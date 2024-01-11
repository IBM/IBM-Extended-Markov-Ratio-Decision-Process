# EMRDP

This repo includes code referenced in the paper _A Rigorous Risk-aware Linear Approach to Extended Markov Ratio Decision Processes with Embedded Learning_ by Alexander Zadorojniy, Takayuki Osogami, and Orit Davidovich to appear in IJCAI 2023.

The code implements the paper's Grid World experiments, including an algorithm to find the theoretically optimal policy for the Grid World environment and Algorithm 3 from the paper.

# Environment

We tested our code on a Python 3.8.0 environment.

# Configs

Our scripts interact with storage, either local or S3. 

Storage specs are provided in a `configs.yaml`. It includes the list of source and target storage folder or bucket names. If necessary, S3 credentials are added under their designated fields.

You'll find acceptable config formats under `./configs/`. Here is the configs format for local storage
```
local:
    buckets:
        inputs: '<inputs-folder>'
        inputs_dest: '<inputs-dest-folder>'
        data: '<data-folder>'
        data_dest: '<data-dest-folder>'
        results: '<solutions-folder>'
        reports: '<reports-folder>'
        logs: '<logs-dest-folder>'
```
and here it is for S3
```
s3:
    buckets:
        inputs: '<inputs-bucket>'
        inputs_dest: '<inputs-dest-bucket>'
        data: '<data-bucket>'
        data_dest: '<data-dest-bucket>'
        results: '<results-bucket>'
        reports: '<reports-bucket>'
    aws_secret_access_key: 'xxxx'
    aws_access_key_id: 'xxxx'
    endpoint_url: 'https://xxx.xxx.xxx'
    region: 'xx-xxxx'
    cloud_service_provider: 'aws'
```

Make a copy of one of the config example yamls under `./configs/` and store it under the root `./`.
```
cd IBM-Extended-Markov-Ratio-Decision-Process
cp configs/ibm_configs_example.yaml configs.yaml
```
Then edit all field values. Note that bucket / folder names **must be distinct**.

Currently, two S3 providers are available under `s3:cloud_service_provider`: either `aws` or `ibm`. The `endpoint_url` is _optional_ for AWS.

# Clone

To run our scripts and notebooks, first clone our repo
```
git clone https://github.com/IBM/IBM-Extended-Markov-Ratio-Decision-Process.git
```
and then 
```
cd IBM-Extended-Markov-Ratio-Decision-Process
```
To run our notebooks, install our code with 
```
pip install .
```

# Experiments

Before you can run our scripts, you must first define your Grid World environment hyper-parameters. These hyper-parameters are defined in `./experiment.json`.

Here are the hyper-parameters we used in the experiments reported in our IJCAI paper
```
{
	"algorithm": {
		"is_theory": [false],
		"is_max": [false]
	},
	"data": {
		"grid": [
			[5, 5]
		],
		"obstacles": [3],
		"noise": [0.1,0.3,1.0],
        "delta": [0.05],
        "epsilon": [0.01],
        "tolerance": [1e-8],
        "sensitivity": [1e-5],
		"discount": [0.95],
        "cost": [5.0],
		"init_state": [
			[0,0]
		]
	},
	"experiment": {
		"N": 150,
		"is_same": false,
		"experiment_id": "experiment01"
	}
}
```

# Scripts

Below you'll find the commands to run the scripts that 
1. generate data from the Grid World environment,
2. apply the EMRDP algorithm with embedded learning (Algorithm 3 in the paper).
```
cd IBM-Extended-Markov-Ratio-Decision-Process
```

To generate data from the Grid World environment and store it *locally*, run
```
python scripts/grid_data.py --configs configs.yaml -l
```
To generate data and upload it to S3, run
```
python scripts/grid_data.py --configs configs.yaml
```
The local version will generate logs when a logs dir is provided in your `configs.yaml`.

Here is how you run the script that applies the EMRDP algorithm against the Grid World datasets that were stored locally
```
python grid_emrdp.py --path_in ~/path/to/data --path_out ~/path/to/results --path_log ~/path/to/logs/dir
```

# Ray Cluster

We implemented the cloud distribution of the EMRDP algorithm using [Ray](https://www.ray.io/ "Ray -- fast and simple distributed computing"). You can run it, assuming you have a K8S cluster already set up. If not, follow the instructions for setting up an IBM OpenShift cluster [here](https://github.com/IBM/doframework/blob/main/docs/openshift.md "OpenShift Cluster") or an AWS K8S cluster [here](https://github.com/IBM/doframework/blob/main/docs/aws.md "AWS K8S Cluster").

Before you run our scripts, make sure you're in the right K8S context. If you have the `kubectl` CLI installed, you can run
```
kubectl config current-context
```
Your context is also available to view on Docker Desktop -> kubernetes -> contexts.

The Ray cluster requires its own cluster yaml. We used the cluster yaml `emrdp.yaml` copied from [another project](https://github.com/IBM/doframework "DOFramework"). Note that `emrdp.yaml` assumes your `configs.yaml` is located under the dir root `./`.

See [here](https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html "Cluster YAML Configuration Options") for instructions on how to define the `Ray` cluster yaml from scratch. 

Start the `Ray` cluster with
```
ray up emrdp.yaml --no-config-cache --yes
```

Run the data generation script [with a log file]
```
ray submit emrdp.yaml scripts/grid_data.py -c configs.yaml --path_log . --no-config-cache
```

To check the log file, first log into the cluster head node
```
ray attach ~/Workspace/emrdp/emrdp.yaml
```
You can `ls -lh` to find the time-stamped name of the log file. For example, read the file with
```
(base) ray@emrdp-cluster-head-jvmbl:~$ cat grid_emrdp_data_2022-12-23_0347.log
```
Exit the cluster head node with
```
(base) ray@emrdp-cluster-head-jvmbl:~$ exit
```
Run the EMRDP algorithm script [without a log file]
```
ray submit emrdp.yaml scripts/grid_emrdp.py -c configs.yaml --no-config-cache
```

## Ray Know-How

To update the Ray cluster yaml after a `emrdp.yaml` change
```
ray up emrdp.yaml --no-config-cache --yes
```

To shutdown the Ray cluster
```
ray down -y emrdp.yaml
```

To monitor Ray cluster autoscaling
```
ray exec emrdp.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
```

To connect to a terminal on the Ray cluster head node
```
ray attach emrdp.yaml
```
Then exit the head node with
```
exit
```

See [here](https://github.com/IBM/doframework/blob/main/docs/openshift.md) for OpenShift-specific instructions on how to view the `Ray` dashboard.

# Issues

--> On `ray up` you get ModuleNotFoundError related to `ray`.
```
pip install -U "ray[default,serve,k8s]"==1.13.0
```

--> When running `grid_emrdp.py`, you get alerts: Received a GOAWAY with error code ENHANCE_YOUR_CALM and debug data equal to "too_many_pings".

Follow [this](https://github.com/ray-project/ray/pull/27769) `Ray` core issue.

--> SSH still not available

Too many CPU / memory resources were requested as max values.
