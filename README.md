# Efficient Task Adaptation by Mixing Discovered Skills; 

This repository implements agents introduced in [Efficient Task Adaptation by Mixing Discovered Skills](https://openreview.net/forum?id=zkG3N8uff3V) paper.

Same importance agent : agents/diayn_same_weight.py
Simple importance agent : agents/diayn_simple_weight.py
DIAYN controller agent : agents/SaP.py
Scratch controller agent : agents/SaP.py (run with 'init_diayn=false' option)

Note that we named our best performing agent DIAYN controller agent as SaP, which stands for 'Skill as persepective'.
This codebase is built on top of the [Unsupervised Reinforcement Learning Benchmark (URLB) codebase](https://github.com/rll-research/url_benchmark). We include agents for all baselines in the `agents` folder.

To obtain pre-trained weight, run the following command:
```sh
python pretrain.py agent=diayn domain=walker experiment=YOUR_EXP_NAME
```

After pretraining, to finetune your agent, run the following command.
```
python finetune.py agent=AGENT_NAME experiment=YOUR_EXP_NAME task=walker_stand
```

Make sure to specify the directory of your saved snapshots referring to .yaml file for each agent.
For example, to run SaP agent, run below code.

```sh
# run SaP agent
python finetune.py agent=SaP experiment=YOUR_EXP_NAME task=walker_stand
```


## Requirements
We assume you have access to a GPU that can run CUDA 10.2 and CUDNN 8. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```sh
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with
```sh
conda activate urlb
```

## Available Domains
We support the following domains.
| Domain | Tasks |
|---|---|
| `walker` | `stand`, `walk`, `run`, `flip` |
| `quadruped` | `walk`, `run`, `stand`, `jump` |
| `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right` |
