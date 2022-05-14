import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from dm_env import specs
from agent.ddpg import Actor, DDPGAgent


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

class SkillMixingActor(Actor):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, pretrained_actor, diayn):
        super().__init__(obs_type, obs_dim, action_dim, feature_dim, hidden_dim)

        self.pretrained_actor = pretrained_actor
        self.diayn = diayn

    def forward(self, obs, std):

        return super().forward(obs, std)

    
class DIAYNasWeightPredictorAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale,
                 update_encoder, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale
        self.update_encoder = update_encoder
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim

        # create actor and critic
        super().__init__(**kwargs)
        self.diayn = DIAYN(self.obs_dim - self.skill_dim, self.skill_dim,
                           kwargs['hidden_dim']).to(kwargs['device'])
        self.diayn.train()

        self.pretrained_actor = self.actor
        self.actor = SkillMixingActor(self.obs_type, self.obs_dim, self.action_dim,
                           self.feature_dim, self.hidden_dim, self.pretrained_actor, self.diayn).to(kwargs['device'])

        self.actor_opt = torch.optim.Adam(self.actor.parameters()) + list(self.diayn.parameters()), lr=self.lr)
    

    def update_actor(self, obs, step):
        metrics = dict()

        obs = torch.as_tensor(obs, device=self.device)  # (B, obs_dim)
        skill_weight = F.softmax(self.diayn(obs), dim=-1).unsqueeze(-1)  # (B, skill_dim, 1)

        obs = obs.unsqueeze(1).repeat(1, self.skill_dim, 1)  # (B, skill_dim, obs_dim)
        skill_list = torch.eye(self.skill_dim, device=self.device).unsqueeze(0)  # (1, skill_dim, skill_dim)
        skill_list = skill_list.repeat(obs.shape[0], 1, 1)  # (B, skill_dim, skill_dim)
        obs_with_skill = torch.cat([obs, skill_list], dim=-1)  # (B, skill_dim, skill_dim + obs_dim)

        inpt = obs_with_skill * skill_weight  # (B, skill_dim, skill_dim + obs_dim)
        h = self.encoder(inpt)

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(h, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        # actor_loss.backward(retain_graph=True)
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def act(self, obs, meta, step, eval_mode):
        """
        meta from passed parameter is useless
        """
        #assert obs.shape[-1] == self.obs_shape[-1]
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0) # (1, obs_dim)
            skill_weight = F.softmax(self.diayn(obs), dim=-1)  # (1, skill_dim)

            obs = obs.repeat(self.skill_dim, 1)  # (skill_dim, obs_dim)
            skill_list = torch.eye(self.skill_dim, device=self.device) # (skill_dim, skill_dim)
            obs_with_skill = torch.cat([obs, skill_list], dim=-1)  # (skill_dim, skill_dim + obs_dim)

            inpt = obs_with_skill * skill_weight.view(self.skill_dim, 1)  # (skill_dim, skill_dim + obs_dim)

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
        utils.hard_update_params(other.actor, self.pretrained_actor)
        utils.hard_update_params(other.diayn, self.diayn)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
