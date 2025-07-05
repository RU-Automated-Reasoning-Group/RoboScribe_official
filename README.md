# RoboScribe

Deep reinforcment learning (RL) methods have significantly advanced robotic system. But due to the complex architectures of neural network models, ensuring their trustworthiness is a considerable challenge. Although Programmatic RL has been developed to improve interpretability with programmatic architecture, their relies on predefined Domain Specific Language (DSL) makes them less flexible and expensive. In our paper, we propose a novel programmatic RL method, RoboScribe, to automatically generate action abstraction and state abstraction predicate refinements given coarse DSL with the purpose to reduce generation cost and increase generalization. Additionally, RoboScribe generate iterative program to support better transferability for task with repeated rountine and arbitrary number of objects. This artifact implements RoboScribe to support these claims. Specifically, the artifact provide the evaluation experiments on Pick&Place-1, Pick&Place-4, Push-3 and Tower-5 to support robot-control program generations (related to the Figure.18 results in the paper). To demonstrate the transferability, this artifact provide experiments to evaluate program —  that generated from Tower-5 environment — on Pyramid and Multi Towers tasks (related to the Figure.20 in the paper). Note that in Figure.18 of the paper, we also present results of Meta-World and PlaceCubesDrawer environment. But due to the specific versions of Python plackages required by these two environments and the limit of time for collecting artifacts, we are unable to support these two environments for now. We will add these two environments in later version.

## Hardware Dependencies

We recommend a hardware with at least 32GB available disk space to run this artifact. Specifically, this artifact contains code of RoboScribe, demonstration date as well as necessary environment simulation occupying around 4GB disk space. The evaluation on environments will generate model checkpoints and log data which could occupy upto 20 GB disk space. In additional, we recommend a GPU with no less that 10GB GPU memory to run the artifact for each environment. And we highly recommend to run the artifact on separate GPU cores for different environments.

## Getting Started Guide

This artifact is implemented with Python3.9. For convenience, we build a docker image for the artifact. Before proceding, please ensure Docker is installed (```sudo docker run hello-world``` will test your installation). If Docker is not installed, please follow the [official installation guide](https://docs.docker.com/get-docker/). This guide was tested using Docker version 20.10.23, but any contemporary Docker version is expected to work.

### Use the Pre-Built Docker Image
Our pre-built docker image could be fetched from Docker Hub:
```
docker pull wyuning/roboscribe:latest
```

To launch a shell in the docker image:
```
docker run -it --gpus all wyuning/roboscribe:latest
```
Once the container successfully starts, you should be automatically placed in the folder called `/RoboScribe` where all our code and scripts are located.

### Basic Test

Run the simple test script:
```
sh basic_test.sh
```

The script will evaluate four environments, including Push-3, Tower-5 and Pick&Place-4. The evaluation will take around 2 minutes and print logs on the terminal. Several warnings might be outputs but could be ignored. A success running will generate 3 log file in ```logs/```:
```
brief_test_push_block3.txt
brief_test_tower5.txt
brief_test_pickplace_multi.txt
```
with the last sentence to be "Test Finish And Success for ..."

Additionally, 3 folders will be created under the directory ```store/```:
```
brief_test_push_block3
brief_test_tower5
brief_test_pickplace_multi
```
Under each folder, a tensorboard log will be stored in ```debug_policy_0``` folder, a predicate graph file ```graph.pkl``` will be stored in ```debug_skill``` and two empty folders — ```debug_skill_fail``` and ```states``` — will be created.

The last sentence and certain files will miss for a **failed test**.

## Step-by-step Instruction

In the artifact, we use ```wandb``` to store data for iterative program training. Please login your wandb account before running the artifact with:
```
wandb login <API_KEY>
```
where ```<API_KEY>``` can be generated with this link: https://wandb.ai/authorize .

In the rest of this section, we introduce the training details of each environment. We highly recommend the reviewers to run Pick&Place-1 and Pick&Place-4 environments first due to the much lower expected running time.

### Pick&Place-1 Environment
To run RoboScribe to generate robot-control program for Pick&Place-1 environment (first figure in the first row of Figure.18 in the paper), run the script as:
```
sh main_v7_block_pickplace1.sh
```

The script will take demonstration ```store/demo_store/custom_pickplacemulti_simple_block1.npy``` as input to synthesize abstract subtask tree and train low-level policies step-by-step. The log file will be stored in ```logs/tree_pickplace_block1.txt```. The candidate abstraction predicate refinement for each step, the result abstract subtask trees and evaluation success rate to achieve speciftic reward are expected to be included in the log file. The abstract subtask tree and low-level polices will be stored in ```store/tree_pickplace_block1/debug_skill```. Specifically, the abstract subtask tree will be stores as file ```store/tree_pickplace_block1/debug_skill/graph.pkl``` and the low-level policies will be stored as file ```store/tree_pickplace_block1/debug_skill/model_<node_id>.pkl```, where ```<node_id>``` refers to the identity number of specific node in the graph. The training data, including evaluation reward during training, training loss, etc, are collected with tensorboard and stored in directories ```store/tree_pickplace_block1/debug_policy_<train_order>``` where ```<train_order>``` refers to the order related low-level policy is trained. 

An execution result is provided in ```logs_backup/tree_pickplace_block1.txt``` and ```store/tree_pickplace_block1```. **The expected time to run this environment is 7 hours with GPU.**

---

### Pick&Place-4 Environment
To run RoboScribe to generate robot-control program for Pick&Place-4 environment (second figure in the first row of Figure.18 in the paper), run the script as:
```
sh main_v7_block_pickplace4.sh
```

The script will take demonstration ```store/demo_store/custom_pickplacemulti_block4_clean.npy``` as input to synthesize abstract subtask tree and train low-level policies step-by-step. Different from Pick&Place-1 environment, RoboScribe generates iterative program for Pick&Place-4 to handle arbitrary number of blocks and repeated routine. The log file will be stored in ```logs/tree_pickplace_block4.txt```. The candidate abstraction predicate refinement for each step, the result abstract subtask trees and evaluation success rate to achieve speciftic reward are expected to be included in the log file. 

The abstract subtask tree and low-level polices will be stored in ```store/tree_pickplace_block4/debug_skill```. Specifically, the abstract subtask tree will be stores as file ```store/tree_pickplace_block4/debug_skill/graph.pkl```, the low-level policies will be stored as file ```store/tree_pickplace_block4/debug_skill/model_<node_id>```, and the generated iterative program will be stored as file ```store/tree_pickplace_block4/debug_skill/iterative_program```. In addition, we retrain low-level policies during iterative program training and store the checkpoints as ```store/tree_pickplace_block4/debug_skill/policy_<policy_id>_<checkpoint_num>.pth```, where ```<policy_id>``` refers to the order of policy in the iterative program and the ```<checkpoint_num>``` refers to the training step where the policy is stored.

An execution result is provided in ```logs_backup/tree_pickplace_block4.txt``` and ```store/tree_pickplace_block4```. **The expected time to run this environment is 20 hours with GPU.**

---

### Push-3 Environment
To run RoboScribe to generate robot-control program for Push-3 environment (fourth figure in the first row of Figure.18 in the paper), run the script as:
```
sh main_v7_block_push.sh
```

The script will take demonstration ```store/demo_store/pushmulti3_coll_debug_2.npy``` as input to synthesize abstract subtask tree and train low-level policies step-by-step. RoboScribe also generates iterative program for Push-3 to handle arbitrary number of blocks and repeated routine. The log file will be stored in ```logs/tree_push_block3.txt```. The candidate abstraction predicate refinement for each step, the result abstract subtask trees and evaluation success rate to achieve speciftic reward are expected to be included in the log file. 

The abstract subtask tree and low-level polices will be stored in ```store/tree_push_block3/debug_skill```. Specifically, the abstract subtask tree will be stores as file ```store/tree_push_block3/debug_skill/graph.pkl```, the low-level policies will be stored as file ```store/tree_push_block3/debug_skill/model_<node_id>```, and the generated iterative program will be stored as file ```store/tree_push_block3/debug_skill/iterative_program```. In addition, we retrain low-level policies during iterative program training and store the checkpoints as ```store/tree_push_block3/debug_skill/policy_<policy_id>_<checkpoint_num>.pth```, where ```<policy_id>``` refers to the order of policy in the iterative program and the ```<checkpoint_num>``` refers to the training step where the policy is stored.

An execution result is provided in ```logs_backup/tree_push_block3.txt``` and ```store/tree_push_block3```. **The expected time to run this environment is 100 hours with GPU.**

---

### Tower-5 Environment

To run RoboScribe to generate robot-control program for Tower-5 environment (first figure in the second row of Figure.18 in the paper), run the script as:
```
sh main_v7_block_tower.sh
```

The script will take demonstration ```store/demo_store/tower_5.npy``` as input to synthesize abstract subtask tree and train low-level policies step-by-step. RoboScribe also generates iterative program for Tower-5 to handle arbitrary number of blocks and repeated routine. The log file will be stored in ```logs/tree_tower5.txt```. The candidate abstraction predicate refinement for each step, the result abstract subtask trees and evaluation success rate to achieve speciftic reward are expected to be included in the log file. 

The abstract subtask tree and low-level polices will be stored in ```store/tree_tower5/debug_skill```. Specifically, the abstract subtask tree will be stores as file ```store/tree_tower5/debug_skill/graph.pkl```, the low-level policies will be stored as file ```store/tree_tower5/debug_skill/model_<node_id>```, and the generated iterative program will be stored as file ```store/tree_tower5/debug_skill/iterative_program```. In addition, we retrain low-level policies during iterative program training and store the checkpoints as ```store/tree_tower5/debug_skill/policy_<policy_id>_<checkpoint_num>.pth```, where ```<policy_id>``` refers to the order of policy in the iterative program and the ```<checkpoint_num>``` refers to the training step where the policy is stored.

An execution result is provided in ```logs_backup/tree_tower5.txt``` and ```store/tree_tower5```. **The expected time to run this environment is 120 hours with GPU.**

---


### PickCubeDrawer Environment

To run RoboScribe to generate robot-control program for PickCubeDrawer environment (first figure in the second row of Figure.18 in the paper), run the script as:
```
sh main_v7_opendrawer.sh
```

The script will take demonstration ```store/demo_store/opendrawer_pickplacecube3_crop.npy``` as input to synthesize abstract subtask tree and train low-level policies step-by-step. RoboScribe also generates iterative program for PickCubeDrawer to handle arbitrary number of blocks and repeated routine. The log file will be stored in ```logs/tree_drawer_pickplacecubemulti.txt```. The candidate abstraction predicate refinement for each step, the result abstract subtask trees and evaluation success rate to achieve speciftic reward are expected to be included in the log file. 

The abstract subtask tree and low-level polices will be stored in ```store/tree_drawer_pickplacecubemulti/debug_skill```. Specifically, the abstract subtask tree will be stores as file ```store/tree_drawer_pickplacecubemulti/debug_skill/graph.pkl```, the low-level policies will be stored as file ```store/tree_drawer_pickplacecubemulti/debug_skill/model_<node_id>```, and the generated iterative program will be stored as file ```store/tree_drawer_pickplacecubemulti/debug_skill/iterative_program```. In addition, we retrain low-level policies during iterative program training and store the checkpoints as ```store/tree_drawer_pickplacecubemulti/debug_skill/policy_<policy_id>_<checkpoint_num>.pth```, where ```<policy_id>``` refers to the order of policy in the iterative program and the ```<checkpoint_num>``` refers to the training step where the policy is stored.

An execution result is provided in ```logs_backup/tree_drawer_pickplacecubemulti.txt``` and ```store/tree_drawer_pickplacecubemulti```. **The expected time to run this environment is 120 hours with GPU.**

---

### Evaluation for Transferability of Tower-5 Policies

We evaluate the transferability of RoboScribe by applied the robot-control program generated on tower-5 environment to Pyramid and Multi-Tower environments (Figure 20 in the paper). For convenience, we provide the checkpoints and iterative program that we trained for tower-5 in the directory ```store_backup/tree_tower5_transfer_load```. Run the script to evaluate the transferability as:
```
sh main_v7_tower_transfer.sh
```

The script will evaluate the program on Pyramid with 4-9 blocks, Multi-Tower with 4-9 blocks and Single Tower with 4-7 blocks. The results will be stored in log file ```logs/tree_tower5_transfer.txt``` and ```store/tree_tower5_transfer/debug_skill/transfer_eval_result.npy```. **The expected time to run this evaluation is 4 hours with GPU**.

---

### Data Visualization
We provide the code to visualize result data by creating Figure.18 and Figure.20 in the paper.

To generate Figure.18 in the paper, we collect the training data of RoboScribe and baselines (deepset, gail, goalgail and bc) in directory ```curve_data_visualize```. User could enter the directory and run ```plot.py``` to generate the figure:
```
cd curve_data_visualize
python plot.py
```
Specifically, if users want to use manually generated data, they can download the evaluation success rate data from the wandb website and upload it to the related directory in ```curve_data_visualize```.

To generate Figure.20 in the paper, we collect the evaluation success rate for single tower, pyramid and multi-tower environments and hard code them in the file ```transfer_data_visualize/plot.py```. User could enter the directory and run the plot file to generate the figure:
```
cd transfer_data_visualize
python transfer_data_visualize/plot.py
```
Similarly, user could add their own log data by modifying the results in function ```tower_robo()``` of ```transfer_data_visualize/plot.py```.


## Reusability Guide
To reuse RoboScribe for a new environment, we provide a step-by-step instructions in this section.

We outline the structure of RoboScribe as below:
```
RoboScribe
|   main_v7.py
|___environments
|   |   general_env.py
|   |   skill_env.py
|   |   [files for environment definition]
|___modules
|   |   dt_learn.py
|   |   reward.py
|   |   skill.py
|___policy
|   |   SAC_LG
|   |   awac.py
|   |   [files for RL policy algorithms]
|___synthesize
|   |   dsl.py
|   |   topdown.py
```

For a custom environment, the code under ```modules```, ```policy``` and ```synthesize``` are reusable, while the code under ```environments``` and ```main_v7.py``` need to be modified to include related content for the new environment.

Specifically, for ```environments```, the definition of new environment need to be added. For example, the directory ```environments/cee_us_env``` refers to the environment directory for Pick&Place-4 environment.

Inside the main file ```main_v7.py```, the functions related to environments need to be modified. For example, for environment Pick&Place-4 environment, the following code should be added:
```
# main_v7.py
def define_env(args, eval=False):
    ...
    elif 'pickmulti' in args.env_name:
        task = 'pickmulti'
        # define environment
        env = GymToGymnasium(FetchPickAndPlaceConstruction(name=args.env_name, sparse=False, shaped_reward=False, num_blocks=args.set_block_num, reward_type='sparse', case = 'PickAndPlace', visualize_mocap=False, simple=True))
        # coarse abstract predicate for the environment
        obs_transit = AbsTransit_Pick_Multi()
```

In addition, demonstration collected in the new environment need to be stored in directory ```store/demo_store```. 

We provide a template script as ```main_v7_custom.sh```. Specifically, the following arguments need to be modified:
```
log_path: name of log file in logs/
skill_path: directory to store policies
policy_path: directory to store policies during predicate refinement
fig_path: directory to store log images
demo_path: file of demonstrations
env_name: name of environment
train_traj_len: trajectory length limit for environment execution during training
eval_traj_len: trajectory length limit for environment execution during evaluation
collect_traj_len: trajectory length limit for environment execution during negative state collection
set_block_num: number of objects inside environment
```
