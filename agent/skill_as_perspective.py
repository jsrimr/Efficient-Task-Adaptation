from collections import OrderedDict
from turtle import forward

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

from agent.ddpg import Actor, Critic, DDPGAgent

class skillActor(Actor):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__(obs_type, obs_dim, action_dim, feature_dim, hidden_dim)
        
    def extract_perspective(self, obs):
        return self.trunk(obs)


class PolicyLayer(nn.Module):
    def __init__(self, obs_type, feature_dim, hidden_dim, action_dim):
        super().__init__()
        policy_layers = [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]
        self.policy = nn.Sequential(*policy_layers)
        self.apply(utils.weight_init)

    def forward(self, augmented_obs, std):
        mu = self.policy(augmented_obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist

class MultiSkillActor(nn.Module):
    # TODO : 어쩌면 double_learning_rate 가 필요할 수도
    def __init__(self, obs_type, obs_dim, feature_dim, hidden_dim, action_dim, **kwargs):
        super().__init__()
        # self.skill_agent_list = skill_agent_list
        self.device = kwargs['device']
        self.skill_agent = skillActor(obs_type, obs_dim, action_dim, feature_dim, hidden_dim).to(self.device)
        self.skill_dim = kwargs['skill_dim']
        self.policy = PolicyLayer(obs_type, feature_dim * self.skill_dim, hidden_dim, action_dim)

    def forward(self, obs, stddev):
        # obs = (B, obs)

        state_with_skill = []
        for meta in range(self.skill_dim):
            skill = torch.zeros([obs.shape[0], self.skill_dim], device=self.device)
            skill[:, meta] = 1.0
            state_with_skill.append(torch.cat([obs, skill], dim=-1).unsqueeze(1))
        
        state_with_skill = torch.cat(state_with_skill, dim=1)  # state_with_skill = (B, skill_dim, concatted_obs)
        perspective = self.skill_agent.extract_perspective(state_with_skill) # perspective = (B, skill_dim, encoded_obs)

        #assert obs.shape[-1] == self.obs_shape[-1]
        return self.policy(perspective.view(perspective.shape[0], -1), stddev)  # (B, action_dim)


class PerspectiveAgent(DDPGAgent):
    def __init__(self,
                 name,
                 reward_free,
                 obs_type,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 num_expl_steps,
                 update_every_steps,
                 stddev_schedule,
                 nstep,
                 batch_size,
                 stddev_clip,
                 init_critic,
                 use_tb,
                 use_wandb,
                 update_encoder,
                 skill_dim=0):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None
        self.update_encoder = update_encoder
        self.meta_dim = skill_dim

        # models
        if obs_type == 'pixels':  # Note : 당분간 encoder 쓸 일은 없다. state 로만 학습할 거임.
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + self.meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + self.meta_dim

        
        self.actor = MultiSkillActor(obs_type, self.obs_dim, feature_dim, hidden_dim, self.action_dim, device=self.device, skill_dim=self.meta_dim).to(device)

        self.critic = Critic(obs_type, obs_shape[0], self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, obs_shape[0], self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        
        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()


    def init_from(self, other):
        # copy parameters over
        # for skill_agent in self.actor.skill_agent_list:
        utils.hard_update_params(other.actor, self.actor)
        # utils.hard_update_params(other.encoder, self.encoder)
        # Note : critic 은 굳이 weight 상속받을 필요 없을듯. pretrain stage 와 전혀 다른 분포를 학습하기 때문.
        # if self.init_critic:
        #     utils.hard_update_params(other.critic.trunk, self.critic.trunk)
