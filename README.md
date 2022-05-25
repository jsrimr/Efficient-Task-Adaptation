# Skill as persepective (SaP)

This codebase is built on top of the [Unsupervised Reinforcement Learning Benchmark (URLB) codebase](https://github.com/rll-research/url_benchmark). We include agents for all baselines in the `agents` folder. Our method `SaP`  is implemented in `agents/SaP.py` and the config is specified in `agents/SaP.yaml`.


To pre-train SaP, run the following command:

```sh
python pretrain.py agent=diayn domain=walker experiment=YOUR_EXP_NAME
```

To finetune SaP, run the following command. Make sure to specify the directory of your saved snapshots with `YOUR_EXP_NAME`.

```sh
python finetune.py agent=SaP experiment=YOUR_EXP_NAME task=walker_stand snapshot_ts=2000000
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