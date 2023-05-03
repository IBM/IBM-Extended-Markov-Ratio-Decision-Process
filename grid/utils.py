import os
import argparse
import pickle
import json
import yaml
from typing import Optional, List
import logging
from pathlib import Path
import random
import string
import numpy as np
from scipy.stats import dirichlet
import ibm_boto3
import boto3

def is_path(path):
    
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"path: {path} is not a valid path.")

def is_file(file):
    
    if os.path.exists(file) and os.path.isfile(file):
        return file
    else:
        raise argparse.ArgumentTypeError(f"path: {file} is not a valid file.")

def get_configs(configs_file: str):

    cf = is_file(configs_file)

    try:
        with open(cf,'r') as handle:
            configs = yaml.safe_load(handle)
        return configs
    except yaml.YAMLError as e:
        print('ERROR ... Could not load configs yaml from {}.'.format(cf))
        raise e   
         
def generate_id(N: int=8) -> str:
    
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))

def setup_logger(logger_name: str, log_file: str, level=logging.DEBUG, to_stream: bool=False):
    
    l = logging.getLogger(logger_name)    
    l.setLevel(level)
    
    #### stop matplotlib from dumping logs
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    l.addHandler(fileHandler)
    
    if to_stream:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        l.addHandler(streamHandler)    

def sample_standard_simplex(d: int, N: int=1) -> np.array:
    
    alpha = np.ones(d)
    
    points = dirichlet(alpha).rvs().flatten() if N==1 else dirichlet(alpha).rvs(N)

    return points        

def get_s3_object(configs):

    s3 = configs['s3']

    assert 'cloud_service_provider' in s3, 'Missing s3:cloud_service_provider in configs.'
    assert s3['cloud_service_provider'] in ['ibm','aws'], 'cloud_service_provider in configs must be either `aws` or `ibm`, you have {}.'.format(s3['cloud_service_provider'])

    if s3['cloud_service_provider'] == 'ibm':

        return ibm_boto3.resource(service_name='s3',
                                region_name=s3['region'],
                                endpoint_url=s3['endpoint_url'],
                                aws_access_key_id=s3['aws_access_key_id'],
                                aws_secret_access_key=s3['aws_secret_access_key'])

    if s3['cloud_service_provider'] == 'aws':

        return boto3.resource(service_name='s3',
                                region_name=s3['region'],
                                aws_access_key_id=s3['aws_access_key_id'],
                                aws_secret_access_key=s3['aws_secret_access_key'])
                                                                         
def load_data(file, **kwargs) -> Optional[dict]:

    bucket_str = kwargs['bucket'] if 'bucket' in kwargs and kwargs['bucket'] else 'data'

    if any(['root' in kwargs,'configs' in kwargs]):

        root = kwargs['root'] if 'root' in kwargs else None
        configs_file = kwargs['configs'] if 'configs' in kwargs else None

        if root:

            path = os.path.join(root,file)

            with open(path, 'rb') as handle:
                data = pickle.load(handle)
                
            return data

        elif configs_file:

            with open(configs_file,'r') as handle:
                try:
                    configs = yaml.safe_load(handle)
                except yaml.YAMLError as e:
                    print('({}) ERROR ... Could not load configs yaml.'.format('root'))
                    raise e        

            s3 = get_s3_object(configs)
            bucket = configs['s3']['buckets'][bucket_str]

            data = s3.Bucket(bucket).Object(file).get()
            data = pickle.loads(data['Body'].read())
            
            return data
        
        else:
            
            print('Nothing loaded: no root dir or configs path found in kwargs.')

            return None            
            
    else:
        
        print('Nothing loaded: no root dir or configs path found in kwargs.')
        
        return None

def load_all(**kwargs) -> List:

    datas = []

    N = kwargs['N'] if 'N' in kwargs else None

    if any(['root_in' in kwargs,'configs' in kwargs]):

        root_in = is_path(kwargs['root_in']) if 'root_in' in kwargs else None
        root_out = is_path(kwargs['root_out']) if 'root_out' in kwargs else None
        configs_file = is_file(kwargs['configs']) if 'configs' in kwargs else None

        if root_in:

            if N:

                files = [[f, Path(f).suffix] for f in Path(root_in).rglob('*') if Path(f).suffix in ['.json','.pickle']][:N]
            
            else:

                files = [[f, Path(f).suffix] for f in Path(root_in).rglob('*') if Path(f).suffix in ['.json','.pickle']]

            for file_in, suffix in files:

                if suffix=='.pickle':

                    try:
                        with open(file_in, 'rb') as handle:
                            data = pickle.load(handle)
                    except pickle.UnpicklingError as e:
                        print('ERROR ... Error occured while unpickling file {} in load_all.'.format(file_in))
                        print(e)
                    except (AttributeError,  EOFError, ImportError, IndexError) as e:
                        print('ERROR ... Error occured while unpickling file {} in load_all.'.format(file_in))
                        print(e)

                if suffix=='.json':

                    try:
                        with open(file_in, 'rb') as handle:
                            data = json.load(handle)
                    except json.JSONDecodeError as e:
                        print('ERROR ... Error occured while decoding file {} in load_all.'.format(file_in))
                        print(e)

                datas.append(data)

                if root_out:

                    file_out = os.path.join(root_out,Path(file_in).name)

                    if suffix=='.pickle':

                        try:    
                            with open(file_out, 'wb') as handle: 
                                pickle.dump(data,handle,protocol=pickle.HIGHEST_PROTOCOL)
                            os.remove(file_in)
                        except pickle.PickleError as e:
                            print('ERROR ... Error occured while encoding file {} in load_all.'.format(file_out))
                            print(e)

                    if suffix=='.json':

                        try:    
                            with open(file_out, 'w') as handle: 
                                json.dump(data,handle)
                            os.remove(file_in)
                        except Exception as e:
                            print('ERROR ... Error occured while encoding file {} in json format in load_all.'.format(file_out))
                            print(e)
                    
                    N = N-1 if N else N

                    if (N is not None) and (N <=0):

                        break

            return datas

        elif configs_file:

            configs = get_configs(configs_file)
            bucket_in = kwargs['bucket_in'] if 'bucket_in' in kwargs else None
            bucket_out = kwargs['bucket_out'] if 'bucket_out' in kwargs else None

            if all([bucket_in, bucket_in in configs['s3']['buckets']]):

                s3 = get_s3_object(configs)
                bucket_input = s3.Bucket(configs['s3']['buckets'][bucket_in])                
                bucket_filter = lambda f: any([f.key.endswith('pickle'),f.key.endswith('json')])

                if N:

                    files = [f for f in bucket_input.objects.all().limit(N) if bucket_filter(f)]

                else:

                    files = [f for f in bucket_input.objects.all() if bucket_filter(f)]

                for f in files:

                    name = f.key
                    obj = bucket_input.Object(name).get()

                    if name.endswith('pickle'):

                        data = pickle.loads(obj['Body'].read())

                    if name.endswith('json'):

                        data = json.loads(obj['Body'].read())

                    datas.append(data)

                    if all([bucket_out, bucket_out in configs['s3']['buckets']]):   

                        bucket_output = s3.Bucket(configs['s3']['buckets'][bucket_out])

                        if name.endswith('pickle'):
                            bucket_output.put_object(Body=pickle.dumps(data,protocol=pickle.HIGHEST_PROTOCOL),Key=name)
                        if name.endswith('json'):
                            bucket_output.put_object(Body=json.dumps(data),Key=name)

                        bucket_input.Object(name).delete()

                return datas

            print('Nothing loaded: no bucket_in in kwargs. bucket_in should be one of the bucket keys in configs[\'s3\'][\'buckets\']].')

            return datas

        else:

            print('Nothing loaded: no legitimate root_in or configs file path found in kwargs.')

            return datas

    else:

        print('Nothing loaded: no legitimate root_in or configs file path found in kwargs.')

        return datas

