import argparse
import logging
from itertools import product
import sys
import os
import pickle
from datetime import datetime
from botocore.exceptions import ClientError

import numpy as np

from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from grid.utils import is_path, is_file, get_configs, generate_id, setup_logger
from grid.policy import get_state, next_state, get_cell_i, get_cell_j
from grid.policy import get_policy, policy_noising, policy_data
from grid.policy import occupation_measure_sanity
from grid.utils import sample_standard_simplex, get_s3_object, load_all
from grid.params import transition_probability_matrix, discounted_probability_matrix, expected_reward_vector, expected_risk_vector
from grid.optimize import optimal_policy

actions = ['U','D','L','R','N']

def generate(**kwargs):
    
    data = kwargs['data'] if 'data' in kwargs else {}
    report = kwargs['report'] if 'report' in kwargs else {}
    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    
    if len(data)==0:

        m = kwargs['grid_rows'] if 'grid_rows' in kwargs and kwargs['grid_rows'] else 3
        n = kwargs['grid_cols'] if 'grid_cols' in kwargs and kwargs['grid_cols'] else 3
        states = [*range(m*n)]

        obstacle_num = kwargs['obstacles'] if 'obstacles' in kwargs and kwargs['obstacles'] else 0
        noise = kwargs['noise'] if 'noise' in kwargs and kwargs['noise'] else 0.2
        discount = kwargs['discount'] if 'discount' in kwargs and kwargs['discount'] else 0.9
        cost = kwargs['cost'] if 'cost' in kwargs and kwargs['cost'] else 1.0

        T = kwargs['T'] if 'T' in kwargs and kwargs['T'] else 10*len(actions)*(m*n)**2
        delta = kwargs['delta'] if 'delta' in kwargs and kwargs['delta'] else 0.05
        epsilon = kwargs['epsilon'] if 'epsilon' in kwargs and kwargs['epsilon'] else 1e-2
        tol = kwargs['tolerance'] if 'tolerance' in kwargs and kwargs['tolerance'] else 1e-6
        sensitivity = kwargs['sensitivity'] if 'sensitivity' in kwargs and kwargs['sensitivity'] else 1e-2

        final = get_state(m-1,n-1,m,n) # bottom right corner
                
        if kwargs['init_state']=='random':
            state_dist = sample_standard_simplex(m*n)
            while True:                
                init_state_index = next_state(states,state_dist)[0]
                if init_state_index != final:
                    break
            init_state = [get_cell_i(init_state_index,m,n),get_cell_j(init_state_index,m,n)]
        else:
            init_state = kwargs['init_state']            
            assert all([isinstance(init_state,(list,tuple)),len(init_state)==2]), \
            'The \'init_state\' field in the input json must contain a list of objects of the form: (1) [i,j], (2) (i,j), or (3) \'random\'.'
            init_state_index = get_state(init_state[0],init_state[1],m,n)
            state_dist = np.eye(N=1,M=m*n,k=init_state_index).flatten() # atomic measure at given state

        states_excluded = list(set(states) - set([final,init_state_index]))
        obstacles = []
        while len(states_excluded)>0:
            obstacle = next_state(states_excluded)[0]
            states_excluded = list(set(states_excluded)-set([obstacle]))
            obstacles.append(obstacle)
            if len(obstacles)==obstacle_num:
                break        

        P = transition_probability_matrix(m,n,delta,tol=tol)
        Q = discounted_probability_matrix(m,n,discount,delta,tol=tol)
        r = -expected_reward_vector(m,n,discount,final,obstacles,P,cost=cost,epsilon=epsilon)
        d = -expected_risk_vector(m,n,epsilon=epsilon)

        # calculating the pareto
        rhos, rewards, risks, success = optimal_policy(states,actions,Q,r,d,state_dist,discount,report,
                                                        sensitivity=sensitivity,tol=tol,logger_name=logger_name)

        if success:
            report['degenerate'] = False
        else:
            report['degenerate'] = True
            raise RuntimeError('optimal_policy success flag=False.')

        A = np.vstack([Q,d.T])
        for rho in rhos:
            occupation_measure_sanity(rho,A,state_dist,discount,report,tol)

        # getting the optimal policy
        ratios = [re/ri for re, ri in zip(rewards, risks)]
        opt_ind = np.array(ratios).argmin()
        rho_opt = rhos[opt_ind]
        p_opt = get_policy(rho_opt, states, actions)
        p_data = policy_noising(p_opt, noise, states, actions, state_dist, discount, P)

        data = {
            'm': m,
            'n': n,
            'final': final,
            'obstacles': obstacles,
            'init_state': init_state,
            'state_dist': state_dist,
            'discount': discount,
            'cost': cost,
            'delta': delta,
            'tol': tol,
            'noise': noise,
            'epsilon': epsilon,
            'sensitivity': sensitivity,
            'P': P,
            'r': r,
            'd': d,
            'p_data': p_data,
            'p_opt': p_opt,
            'pareto_rewards': rewards,
            'pareto_risks': risks,
            'pareto_rhos': rhos,
        }

    else:

        m = data['m']
        n = data['n']
        states = [*range(m*n)]
        init_state = data['init_state']
        init_state_index = get_state(init_state[0],init_state[1],m,n)
        T = data['df'].size
        P = data['P']
        r = data['r']
        d = data['d']
        p_data = data['p_data']

    df = policy_data(p_data,init_state_index,T,states,actions,m,n,P,r,d)
    now = datetime.now().strftime('%Y-%m-%d-%H%M')
    policy_data_id = generate_id()

    data = {**{k:v for k,v in data.items() if k not in ['df','timestamp','policy_data_id','data']},**{'df': df, 'timestamp': now, 'policy_data_id': policy_data_id}}

    return data

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_in', type=is_path, help="Absolute path to local directory to load experiment input config.")
    parser.add_argument('--path_out', type=is_path, help="Absolute path to local directory to store generated data objects.")
    parser.add_argument('--path_archive', type=is_path, help="Absolute path to local directory to archive experiment input config.")
    parser.add_argument('--path_log', type=is_path, help="Absolute path to local directory to write log files.")
    parser.add_argument('--path_report', type=is_path, help="Absolute path to local directory to write data generation report.")
    parser.add_argument('-c', '--configs', type=is_file, help="Absolute path to configs yaml for S3 resources and credentials.")

    args = parser.parse_args()

    args_dict = args.__dict__
    kwargs = {k: v for k,v in args_dict.items()}

    if kwargs['path_log']:

        now = datetime.now().strftime('%Y-%m-%d_%H%M')
        log_file = 'grid_emrdp_data_{}.log'.format(now)
        log_path = os.path.join(kwargs['path_log'],log_file)
        logger_name = 'grid_emrdp_data_log'
        setup_logger(logger_name, log_path)

    else:

        logger_name = None
    
    assert any([all([kwargs['path_in'],kwargs['path_out']]),kwargs['configs']]),\
    'Args should include either an absolute paths to local source and target dirs or a configs yaml for Cloud Object Storage.'
    
    is_local = all([kwargs['path_in'],kwargs['path_out']])

    if is_local:

        if kwargs['path_archive']:
            inputs = load_all(root_in=kwargs['path_in'],root_out=kwargs['path_archive'])
        else:
            inputs = load_all(root_in=kwargs['path_in'])
        is_report = all([kwargs['path_report']])

    else:

        inputs = load_all(configs=kwargs['configs'],bucket_in='inputs',bucket_out='inputs_dest')
        configs = get_configs(kwargs['configs'])
        s3 = get_s3_object(configs)
        target_bucket = configs['s3']['buckets']['data']
        is_report = 'reports' in configs['s3']['buckets']

    assert inputs is not None, 'No input extracted.'

    report = {}

    for i,input in enumerate(inputs):

        N = input['experiment']['N'] if 'N' in input['experiment'] else 0
        assert N>0, 'Nothing to generate! Specify a number of datasets to generate under \'experiment\', e.g. \'N\': 2 for two generated datasets.'
        max_tries = 10*N
        is_same = input['experiment']['is_same'] if 'is_same' in input['experiment'] else False

        if logger_name:
            log = logging.getLogger(logger_name)
            log.info('Extracted input config for experimentID={}. Will generate N={} datasets per setup. is_same={}.'.format(input['experiment']['experiment_id'],N,is_same))

        test_setups = [list(product([k],v)) for k,v in {**input['algorithm'],**input['data']}.items()]
        test_configs = [dict(l) for l in list(product(*test_setups))]

        if 'experiment_id' in input['experiment']:
            din = report[input['experiment']['experiment_id']] = {}
        else:
            din = report[i] = {}

        for c, config in enumerate(test_configs):

            din[c] = {}

            cntr, tries = 0, 0

            while all([cntr<N, tries<max_tries]):

                tries += 1

                d = din[c][tries] = {}                

                if logger_name:
                    log = logging.getLogger(logger_name)
                    log.info('Generating the {}th policy dataset on the {}th try for experimentID={} in configuration:\n{}.'.format(cntr+1,tries,input['experiment']['experiment_id'],config))

                try:

                    config_adj = {
                        **{k: v for k,v in config.items() if k!='grid'},
                        **{'grid_rows': config['grid'][0],'grid_cols': config['grid'][1]},
                        'report': d
                        }

                    if cntr==0:

                        data = generate(**config_adj)

                    else:

                        if is_same:

                            config_adj = {'data': config_adj['data'], 'report': d} if 'data' in config_adj and isinstance(config_adj['data'],dict) and len(config_adj['data'])>0 else {'data': data, 'report': d}

                            data = generate(**config_adj)

                        else:

                            data = generate(**config_adj)

                    data = {**data, **input['experiment'], **{k:v for k,v in config_adj.items() if all([k not in data.keys(), k != 'data'])}}
                    p_file = 'grid_world_experimentID={}_dataID={}_m={}_n={}_noise={}.pickle'.format(data['experiment_id'],
                                                                                                    data['policy_data_id'],
                                                                                                    data['m'],
                                                                                                    data['n'],
                                                                                                    int(100 * data['noise']))
                
                    if logger_name:
                        log = logging.getLogger(logger_name)
                        log.info('Writing {}th policy dataset for experimentID={} to file {}.'.format(cntr+1,input['experiment']['experiment_id'],p_file))

                    if is_local:

                        p_path = os.path.join(kwargs['path_out'],p_file)

                        with open(p_path, 'wb') as file: pickle.dump(data,file,protocol=pickle.HIGHEST_PROTOCOL)

                        if logger_name:
                            log = logging.getLogger(logger_name)
                            log.info('Uploaded file {} to dir {}.'.format(p_file,kwargs['path_out']))

                    else:

                        s3.Bucket(target_bucket).put_object(Body=pickle.dumps(data,protocol=pickle.HIGHEST_PROTOCOL),Key=p_file)

                        if logger_name:
                            log = logging.getLogger(logger_name)
                            log.info('Uploaded file {} to bucket {}.'.format(p_file,target_bucket))

                    cntr += 1

                except KeyError as e:                    

                    if logger_name:
                        log = logging.getLogger(logger_name)
                        log.error('KeyError')
                        log.error(e)
                    else:
                        print(e)
                    continue

                except ClientError as e:

                    if logger_name:
                        logger = logging.getLogger(logger_name) 
                        log.error('IBM Cloud S3 connection error.')
                        logger.error(e)
                    else:
                        print(e)
                    continue
            
                except RuntimeError as e:

                    if logger_name:
                        logger = logging.getLogger(logger_name) 
                        logger.error('Runtime error.')
                        logger.error(e)

                    continue

                except AssertionError as e:

                    if logger_name:
                        logger = logging.getLogger(logger_name) 
                        logger.error('Assertion error.')
                        logger.error(e)

                    continue
                
                except Exception as e:                    

                    if logger_name:
                        log = logging.getLogger(logger_name)
                        log.debug('Theoretical optimization failed on the {}th try, perhaps due to numerical volatily. Generated {} datasets so far. Re-trying.'.format(tries,cntr+1))
                        log.debug(e)
                    else:
                        print(e)
                    continue

    if is_report:

        now = datetime.now().strftime('%Y-%m-%d-%H%M')
        if 'experiment_id' in input['experiment']:
            r_file = 'grid_world_experimentID={}_dataReport_{}.pickle'.format(input['experiment']['experiment_id'],now)
        else:
            r_file = 'grid_world_dataReport{}_{}.pickle'.format(i,now)

        if is_local:

            r_path = os.path.join(kwargs['path_report'],r_file)

            with open(r_path, 'wb') as file: pickle.dump(din,file,protocol=pickle.HIGHEST_PROTOCOL)

            if logger_name:
                log = logging.getLogger(logger_name)
                log.info('Uploaded report file {} to dir {}.'.format(r_file,kwargs['path_report']))

        else:

            report_bucket = configs['s3']['buckets']['reports']
            s3.Bucket(report_bucket).put_object(Body=pickle.dumps(din,protocol=pickle.HIGHEST_PROTOCOL),Key=r_file)

            if logger_name:
                log = logging.getLogger(logger_name)
                log.info('Uploaded report file {} to bucket {}.'.format(r_file,report_bucket))

    else:

        print(din)
