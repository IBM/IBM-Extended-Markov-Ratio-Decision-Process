import sys
import os
import ray
import pickle
import argparse
import logging
from datetime import datetime
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from grid.utils import is_path, is_file, setup_logger, get_s3_object, get_configs, load_all
from emrdp.algorithm import EMRDPAlgorithm, Policy

def optimize(data, n_q_iteration: int=100, logger_name: Optional[str]=None):

    emr = EMRDPAlgorithm(
            data['state_dist'],
            data['P'], 
            data['tol'], 
            data['r'], 
            data['d'],
            data['discount'],
            data['df'],
            ['state_i', 'state_j'],
            ['action'], 
            ['reward'],
            ['risk'],
            is_theory=data['is_theory'],
            is_max=data['is_max'],
            n_q_iteration=n_q_iteration, 
            logger_name=logger_name
        )

    candidates_list = emr.run()
    _, opt_policy = emr.return_optimal_ratio(candidates_list)

    now = datetime.now().strftime('%Y-%m-%d-%H%M')

    result = {
        'rewards_alg': emr.rewards_alg,
        'risks_alg': emr.risks_alg,
        'ratios_alg': emr.ratios_alg,
        'rewards_alg_true': emr.rewards_alg_true,
        'risks_alg_true': emr.risks_alg_true,
        'ratios_alg_true': emr.ratios_alg_true,
        'policy_opt': opt_policy,
        'theory': emr.theory, 
        'timestamp': now
    }

    result = {**result,**{k:v for k,v in data.items() if k not in ['df','timestamp','data']}}

    return result

if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('--path_in', type=is_path, help="Absolute path to local directory to load data objects.")
        parser.add_argument('--path_out', type=is_path, help="Absolute path to local directory to store result objects.")
        parser.add_argument('--path_archive', type=is_path, help="Absolute path to local directory to archive processed data objects.")
        parser.add_argument('--path_log', type=is_path, help="Absolute path to local directory to write log files.")
        parser.add_argument('-c', '--configs', type=is_file, help="Absolute path to configs yaml for S3 resources and credentials.")

        args = parser.parse_args()

        args_dict = args.__dict__
        kwargs = {k: v for k, v in args_dict.items()}

        if kwargs['path_log']:

            now = datetime.now().strftime('%Y-%m-%d_%H%M')
            log_file = 'grid_emrdp_algorithm_{}.log'.format(now)
            log_path = os.path.join(kwargs['path_log'],log_file)
            logger_name = 'grid_emrdp_algorithm_log'
            setup_logger(logger_name, log_path)

        else:

            logger_name = None

        assert any([all([kwargs['path_in'],kwargs['path_out']]),kwargs['configs']]),\
        'Args should include either an absolute paths to local source and target dirs or a configs yaml for Cloud Object Storage.'

        is_local = all([kwargs['path_in'],kwargs['path_out']])

        now = datetime.now().strftime('%Y-%m-%d-%H%M')
        print('Started at {}.'.format(now))        

        if is_local:

            if kwargs['path_archive']:
                data_obj_list = load_all(root_in=kwargs['path_in'],root_out=kwargs['path_archive'])
            else:
                data_obj_list = load_all(root_in=kwargs['path_in'])

        else:

            data_obj_list = load_all(configs=kwargs['configs'],bucket_in='data',bucket_out='data_dest')

        assert all([data_obj_list,len(data_obj_list)>0]), 'No data objects loaded.'

        ray.init(ignore_reinit_error=True)

        for data in data_obj_list:

            now = datetime.now().strftime('%Y-%m-%d-%H%M')
            print('Producing optimal policy for dataID={} at {} ...'.format(data['policy_data_id'],now))

            result = optimize(data,logger_name=logger_name)            
            result_file = 'grid_world_experimentID={}_dataID={}_result_{}_m={}_n={}_noise={}.pickle'.format(
                result['experiment_id'],
                result['policy_data_id'], now,
                result['m'],
                result['n'],
                int(100 * result['noise']))

            if logger_name:
                log = logging.getLogger(logger_name)
                log.info('Writing optimal poilcy result for experimentID={} dataID={} to file {}.'.format(result['experiment_id'],result['policy_data_id'],result_file))

            if is_local:

                p_path = os.path.join(kwargs['path_out'], result_file)

                with open(p_path, 'wb') as file: pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)

                if logger_name:
                    log = logging.getLogger(logger_name)
                    log.info('Uploaded file {} to dir {}.'.format(result_file,kwargs['path_out']))

            else:

                configs = get_configs(kwargs['configs'])
                s3 = get_s3_object(configs)
                target_bucket = configs['s3']['buckets']['results']                
                s3.Bucket(target_bucket).put_object(Body=pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL),Key=result_file)

                if logger_name:
                    log = logging.getLogger(logger_name)
                    log.info('Uploaded file {} to bucket {}.'.format(result_file,target_bucket))

        now = datetime.now().strftime('%Y-%m-%d-%H%M')
        print('Finished at {}.'.format(now))
            

