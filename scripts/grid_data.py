import argparse
import logging
import time
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
    is_eliminate = kwargs['is_eliminate'] if 'is_eliminate' in kwargs else False
    is_recalculate = kwargs['is_recalculate'] if 'is_recalculate' in kwargs else False
    is_verbose = kwargs['is_verbose'] if 'is_verbose' in kwargs else False
    logger_name = kwargs['logger_name'] if 'logger_name' in kwargs else None
    sensitivity = kwargs['sensitivity'] if 'sensitivity' in kwargs and kwargs['sensitivity'] else 1e-2
    
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
            state_dist = np.eye(N=1,M=m*n,k=init_state_index).flatten() # atomic measure at init state

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
        tic = time.perf_counter()
        rhos, rewards, risks, success = optimal_policy(states,actions,Q,r,d,state_dist,discount,
                                                       report=report,is_eliminate=is_eliminate,
                                                       sensitivity=sensitivity,tol=tol,
                                                       logger_name=logger_name if is_verbose else None)
        toc = time.perf_counter()
        if logger_name:
            log = logging.getLogger(logger_name)
            log.debug(f'generate: calculated optimal policy in {(toc-tic)/60:.2f} mins.')

        if success:
            report['degenerate'] = False
        else:
            report['degenerate'] = True
            raise RuntimeError('generate: optimal policy calc success=False.')

        A = np.vstack([Q,d.T])
        for rho in rhos:
            occupation_measure_sanity(rho,A,state_dist,discount,report,tol*100)

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
            'Q': Q,
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

        if is_recalculate: # re-calculating the pareto       
               
            discount = data['discount']
            delta = data['delta']
            tol = data['tol']
            noise = data['noise']            
            state_dist = data['state_dist']            
            Q = data['Q'] if 'Q' in data else discounted_probability_matrix(m,n,discount,delta,tol=tol) # no Q in old data generation results
            policy_data_id = data['policy_data_id']

            tic = time.perf_counter()
            rhos, rewards, risks, success = optimal_policy(states,actions,Q,r,d,state_dist,discount,
                                                           report=report,is_eliminate=is_eliminate,
                                                           sensitivity=sensitivity,tol=tol,
                                                           logger_name=logger_name if is_verbose else None)
            toc = time.perf_counter()
            if logger_name:
                log = logging.getLogger(logger_name)
                log.debug(f'generate: recalculated optimal policy in {(toc-tic)/60:.2f} mins.')

            if success:
                report['degenerate'] = False
            else:
                report['degenerate'] = True
                raise RuntimeError('generate: optimal policy calc success=False.')

            A = np.vstack([Q,d.T])
            for rho in rhos:
                occupation_measure_sanity(rho,A,state_dist,discount,report,tol*100)

            # getting the optimal policy
            ratios = [re/ri for re, ri in zip(rewards, risks)]
            opt_ind = np.array(ratios).argmin()
            rho_opt = rhos[opt_ind]
            p_opt = get_policy(rho_opt, states, actions)
            p_data = policy_noising(p_opt, noise, states, actions, state_dist, discount, P)
            data['recalc_policy_data_id'] = policy_data_id
            data['p_data'] = p_data
            data['p_opt'] = p_opt
            data['pareto_rewards'] = rewards
            data['pareto_risks'] = risks
            data['pareto_rhos'] = rhos

        else:
            p_data = data['p_data']

    df = policy_data(p_data,init_state_index,T,states,actions,m,n,P,r,d)
    now = datetime.now().strftime('%Y-%m-%d-%H%M')
    policy_data_id = generate_id()

    data = {**{k:v for k,v in data.items() if k not in ['df','timestamp','policy_data_id','data']},**{'df': df, 'timestamp': now, 'policy_data_id': policy_data_id}}

    return data

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--local', action='store_true', default=False, help="experiment input config in local storage (true) or S3 (false). defaults to S3.")
    parser.add_argument('-r', '--recalculate', action='store_true', default=False, help="recalculate the optimal policy for data files in local data folder. defaults to False.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="run data generation in verbose mode.")
    parser.add_argument('-c', '--configs', type=is_file, help="Absolute path to configs yaml for S3 resources and credentials.")

    args = parser.parse_args()

    args_dict = args.__dict__
    kwargs = {k: v for k,v in args_dict.items()}
    assert all([kwargs['configs']]),'You must provide an absolute path to your configs yaml: --configs path/to/configs.yaml'
    configs = get_configs(kwargs['configs'])

    is_local = kwargs['local']
    is_verbose = kwargs['verbose']
    is_recalculate = kwargs['recalculate']
    is_log = 'logs' in configs['local']['buckets'] if is_local else False
    is_report = 'reports' in configs['local']['buckets'] if is_local else 'reports' in configs['s3']['buckets']
    logger_name = None

    if is_log:
        if is_local: # logger defined in local run 
            now = datetime.now().strftime('%Y-%m-%d_%H%M')
            log_file = 'grid_emrdp_data_{}.log'.format(now)
            log_path = os.path.join(configs['local']['buckets']['logs'],log_file)
            print(f'See logs in: {log_path}')
            logger_name = 'grid_emrdp_data_log'
            setup_logger(logger_name, log_path)

    if logger_name:
        log = logging.getLogger(logger_name)
        log.info(f'grid_data: is_local={is_local}, is_log={is_log}, is_report={is_report}, is_verbose={is_verbose}, ')
    else:
        print(f'grid_data: is_local={is_local}, is_log={is_log}, is_report={is_report}, is_verbose={is_verbose}')

    source = 'data' if is_recalculate else 'inputs'
    archive = 'data_dest' if is_recalculate else 'inputs_dest'
    target = 'data'

    if is_local:
        inputs = load_all(root_in=configs['local']['buckets'][source],root_out=configs['local']['buckets'][archive])       
        path_out = configs['local']['buckets'][target]
    else:
        inputs = load_all(configs=kwargs['configs'],bucket_in=source,bucket_out=archive)        
        s3 = get_s3_object(configs)
        target_bucket = configs['s3']['buckets'][target]

    assert inputs is not None, 'No input extracted.'

    if logger_name:
        log = logging.getLogger(logger_name)
        log.info('grid_data: number of experiment input files extracted={}.'.format(len(inputs)))
    else:
        print('grid_data: number of experiment input files extracted={}.'.format(len(inputs)))

    report = {}

    if is_recalculate:

        is_eliminate = False

        din = report['recalculate'] = {}

        for i, input in enumerate(inputs):

            policy_data_id = input['policy_data_id'] if 'policy_data_id' in input else None
            assert policy_data_id, 'Nothing to generate! No policy ID in input.'
            N = 1
            max_tries = 10*N

            if logger_name:
                log = logging.getLogger(logger_name)
                log.info('grid_data: working on recalculating optimal policy and generating data for previous data_policy_id={}.'.format(policy_data_id))
            else:
                print('grid_data: working on recalculating optimal policy and generating data for previous data_policy_id={}.'.format(policy_data_id))

            din[policy_data_id] = {}

            cntr, tries = 0, 0

            while all([cntr<N, tries<max_tries]):

                tries += 1

                d = din[policy_data_id][tries] = {}                
                
                try:

                    config = {
                        'data': input,
                        'is_recalculate': is_recalculate,
                        'is_eliminate': is_eliminate,
                        'logger_name': logger_name,
                        'is_verbose': is_verbose,
                        'report': d
                        }
            
                    data = generate(**config)

                    data = {**data, **{k:v for k,v in config.items() if all([k not in data.keys(), k not in ['data','logger_name','report']])}}
                    p_file = 'grid_world_recaculate_policyID_{}_to_dataID={}_m={}_n={}_noise={}.pickle'.format(
                        policy_data_id,data['policy_data_id'],
                        data['m'],
                        data['n'],
                        int(100 * data['noise'])
                    )
                
                    if is_local:

                        p_path = os.path.join(path_out,p_file)

                        with open(p_path, 'wb') as file: pickle.dump(data,file,protocol=pickle.HIGHEST_PROTOCOL)

                        if logger_name:
                            log = logging.getLogger(logger_name)
                            log.info('grid_data: uploaded file {} to dir {}.'.format(p_file,path_out))

                    else:

                        s3.Bucket(target_bucket).put_object(Body=pickle.dumps(data,protocol=pickle.HIGHEST_PROTOCOL),Key=p_file)

                        if logger_name:
                            log = logging.getLogger(logger_name)
                            log.info('grid_data: uploaded file {} to bucket {}.'.format(p_file,target_bucket))

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
                        logger.error('Runtime error:')
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
                        log.debug('grid_data: optimization failed on the {}th try. Generated {} datasets so far. Re-trying.'.format(tries,cntr+1))
                        log.debug(e)
                    else:
                        print(e)
                    continue

        if is_report:

            now = datetime.now().strftime('%Y-%m-%d-%H%M')
            r_file = 'grid_world_data_report_{}_{}.pickle'.format('recalculate',now)

            if is_local:

                r_path = os.path.join(configs['local']['buckets']['reports'],r_file)

                with open(r_path, 'wb') as file: pickle.dump(din,file,protocol=pickle.HIGHEST_PROTOCOL)

                if logger_name:
                    log = logging.getLogger(logger_name)
                    log.info('grid_data: uploaded report file {} to dir {}.'.format(r_file,configs['local']['buckets']['reports']))

            else:

                report_bucket = configs['s3']['buckets']['reports']
                s3.Bucket(report_bucket).put_object(Body=pickle.dumps(din,protocol=pickle.HIGHEST_PROTOCOL),Key=r_file)

                if logger_name:
                    log = logging.getLogger(logger_name)
                    log.info('grid_data: uploaded report file {} to bucket {}.'.format(r_file,report_bucket))

        else:

            print(din)            

    else:

        for i,input in enumerate(inputs):

            N = input['experiment']['N'] if 'N' in input['experiment'] else 0
            assert N>0, 'Nothing to generate! Specify a number of datasets to generate under \'experiment\', e.g. \'N\': 2 for two generated datasets.'
            max_tries = 10*N
            is_same = input['experiment']['is_same'] if 'is_same' in input['experiment'] else False

            if logger_name:
                log = logging.getLogger(logger_name)
                log.info('grid_data: extracted input config for experimentID={}. Will generate N={} datasets per setup.'.format(input['experiment']['experiment_id'],N))
                log.info('grid_data: is_same={}.'.format(is_same))
            else:
                print('grid_data: extracted input config for experimentID={}. Will generate N={} datasets per setup.'.format(input['experiment']['experiment_id'],N))
                print('grid_data: parameters:- is_same={}.'.format(is_same))

            test_setups = [list(product([k],v)) for k,v in {**input['algorithm'],**input['data']}.items()]
            test_configs = [dict(l) for l in list(product(*test_setups))]

            if 'experiment_id' in input['experiment']:
                din = report[input['experiment']['experiment_id']] = {}
            else:
                din = report[i] = {}

            for c, config in enumerate(test_configs):

                if logger_name:
                    log = logging.getLogger(logger_name)
                    log.info('='*75)
                    log.info('grid_data: running experimentID={} in configuration:\n{}.'.format(input['experiment']['experiment_id'],config))
                    log.info('='*75)
                else:
                    print('='*50)
                    print('grid_data: running experimentID={} in configuration:\n{}.'.format(input['experiment']['experiment_id'],config))
                    print('='*50)

                din[c] = {}

                cntr, tries = 0, 0

                while all([cntr<N, tries<max_tries]):

                    tries += 1

                    d = din[c][tries] = {}                

                    if logger_name:
                        log = logging.getLogger(logger_name)
                        log.info('grid_data: generating the {}-th policy dataset on the {}-th try.'.format(cntr+1,tries))

                    try:

                        config_adj = {
                            **{k: v for k,v in config.items() if k!='grid'},
                            **{'grid_rows': config['grid'][0],'grid_cols': config['grid'][1]},
                            'logger_name': logger_name if is_verbose else None,
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
                    
                        if is_local:

                            p_path = os.path.join(path_out,p_file)

                            with open(p_path, 'wb') as file: pickle.dump(data,file,protocol=pickle.HIGHEST_PROTOCOL)

                            if logger_name:
                                log = logging.getLogger(logger_name)
                                log.info('grid_data: uploaded file {} to dir {}.'.format(p_file,path_out))

                        else:

                            s3.Bucket(target_bucket).put_object(Body=pickle.dumps(data,protocol=pickle.HIGHEST_PROTOCOL),Key=p_file)

                            if logger_name:
                                log = logging.getLogger(logger_name)
                                log.info('grid_data: uploaded file {} to bucket {}.'.format(p_file,target_bucket))

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
                            logger.error('Runtime error:')
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
                            log.debug('grid_data: optimization failed on the {}th try. Generated {} datasets so far. Re-trying.'.format(tries,cntr+1))
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

                    r_path = os.path.join(configs['local']['buckets']['reports'],r_file)

                    with open(r_path, 'wb') as file: pickle.dump(din,file,protocol=pickle.HIGHEST_PROTOCOL)

                    if logger_name:
                        log = logging.getLogger(logger_name)
                        log.info('grid_data: uploaded report file {} to dir {}.'.format(r_file,configs['local']['buckets']['reports']))

                else:

                    report_bucket = configs['s3']['buckets']['reports']
                    s3.Bucket(report_bucket).put_object(Body=pickle.dumps(din,protocol=pickle.HIGHEST_PROTOCOL),Key=r_file)

                    if logger_name:
                        log = logging.getLogger(logger_name)
                        log.info('grid_data: uploaded report file {} to bucket {}.'.format(r_file,report_bucket))

            else:

                print(din)
