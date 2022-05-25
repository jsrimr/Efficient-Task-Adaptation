import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from agent.ddpg import DDPGAgent


class DIAYN(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim):
        super().__init__()
        self.skill_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, skill_dim))

        self.apply(utils.weight_init)

    def forward(self, obs):
        skill_pred = self.skill_pred_net(obs)
        return skill_pred



class SkillasPerspectiveAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale,
                 update_encoder, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale
        self.update_encoder = update_encoder
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim
        self.init_diayn = kwargs.pop('init_diayn')

        # create actor and critic
        super().__init__(**kwargs)
        self.diayn = DIAYN(self.obs_dim - self.skill_dim, self.skill_dim,
                           kwargs['hidden_dim']).to(kwargs['device'])
        self.diayn.train()
        self.actor_opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.diayn.parameters()), lr=self.lr)
    

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment and encode : aug_and_encode has no meaning when using 'state' type
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        obs = self.mix_skill_obs(obs)
        next_obs = self.mix_skill_obs(next_obs)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def mix_skill_obs(self, obs):
        """
        (B, obs_dim) => (B, skill_dim + obs_dim)
        """
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(1)  # (B, 1, obs_dim)
        skill_weight = F.softmax(self.diayn(obs), dim=-1) # (B, 1, skill_dim)

        obs = obs.repeat(1, self.skill_dim, 1)  # (B, skill_dim, obs_dim)
        
        skill_list = torch.eye(self.skill_dim, device=self.device).unsqueeze(0)  # (1, skill_dim, skill_dim)
        skill_list = skill_list.repeat(obs.shape[0], 1, 1)  # (B, skill_dim, skill_dim)
        state_with_skill = torch.cat([obs, skill_list], dim=-1)  # (B, skill_dim, skill_dim + obs_dim)

        processed = state_with_skill * skill_weight.permute(0,2,1)  # (B, skill_dim, skill_dim + obs_dim)

        return processed.sum(dim=1)  # (B, skill_dim + obs_dim)

    def act(self, obs, meta, step, eval_mode):
        """
        meta from passed parameter is useless
        """
        #assert obs.shape[-1] == self.obs_shape[-1]
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0) # (1, obs_dim)
        
        inpt = self.mix_skill_obs(obs)
        h = self.encoder(inpt)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(h, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)

        return action.cpu().numpy()[0]
        

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_diayn:
            print("load DIAYN WEIGHT!!")
            utils.hard_update_params(other.diayn, self.diayn)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
