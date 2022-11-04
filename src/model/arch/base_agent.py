import torch
import torch.nn as nn
import numpy as np
from ..builder import ARCHS
from ..builder import build_arch
from ..head import BaseLayer
from .base import BaseArch


@ARCHS.register_module()
class BaseAgent(BaseArch):
    def __init__(self, agent, greedy_range=[0.01, 1.0], greedy_end=1000, update_interval=10, gamma=0.99,
                 losses=dict(type='TorchLoss', loss_name='MSELoss', loss_weight=1.0),
                 **kwargs):
        super(BaseAgent, self).__init__(**kwargs)
        assert agent is not None, 'agent is not defined'
        self.name = 'BaseAgent'
        # pull out env and replay buffer from dataloader
        self.env = self.dataloader.dataset.env
        self.n_actions = self.env.action_space.n
        self.update_interval = update_interval
        # build agent
        agent.dataloader = self.dataloader
        agent.backbone.in_channels = self.env.observation_space.shape[0]
        agent.head.num_classes = self.n_actions
        self.main_agent = build_arch(agent)
        self.target_agent = build_arch(agent)
        # build proxy for loss function
        self.losses = BaseLayer(losses)
        # epsilon greedy policy
        self.gamma = gamma
        self.greedy_range = greedy_range
        self.greedy_end = greedy_end
        # state reset
        self.state = self.env.reset()
        self.state = self.state[0] if isinstance(self.state, tuple) else self.state
        self.total_reward = 0
        self.episode_reward = 0
        # clean up
        self.cleanup()

    def forward(self, x, label=None):
        return self.forward_test(x, label=None)

    def get_epsilon(self):
        # get epsilon for the epsilon greedy policy
        if self.trainer.global_step > self.greedy_end:
            return self.greedy_range[0]
        return self.greedy_range[1] - (self.trainer.global_step / self.greedy_end) * (self.greedy_range[1] - self.greedy_range[0])

    def get_action(self, model, epsilon=None):
        # get action from the agent based on the epsilon greedy policy
        if epsilon is None:
            epsilon = self.epsilon
        if torch.rand(1) < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.from_numpy(np.array([self.state])).to(self.device)
            action = int(model.forward_test(state)['output'].argmax(dim=-1).item())
        return action

    @torch.no_grad()
    def forward_step(self, model, epsilon):
        model.eval()
        action = self.get_action(model, epsilon)

        # do step in the environment
        new_state, reward, terminated, truncated, info = self.env.step(action)
        new_state = new_state[0] if isinstance(new_state, tuple) else new_state

        done = terminated or truncated

        exp = {'state': self.state, 'action': action, 'reward': reward, 'done': done, 'new_state': new_state}
        self.trainer.datamodule.trainset.buffer.append(exp)

        self.state = new_state
        if done:
            self.state = self.env.reset()
            self.state = self.state[0] if isinstance(self.state, tuple) else self.state
        model.train()
        return reward, done

    def forward_train(self, x, label=None):
        # play step in the environment
        epsilon = self.get_epsilon()
        reward, done = self.forward_step(self.main_agent, epsilon)
        self.total_reward += reward
        # play step in the memory buffer
        losses = self.experience_step(x, label=None)
        # stop criterion has been reached
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
        losses.update({'reward': reward, 'total_reward': self.total_reward})
        self.soft_update()
        return losses

    def experience_step(self, x, label=None):
        # train step by experience replay
        state, action, reward, done, new_state = x['state'], x['action'], x['reward'], x['done'], x['new_state']
        state_action_values = self.main_agent.forward_test(state)['output'].gather(1, action.long().unsqueeze(-1)).squeeze(-1)

        # update the target agent
        with torch.no_grad():
            next_state_values = self.target_agent.forward_test(new_state)['output'].max(1)[0].detach()
            next_state_values[done] = 0.0
            next_state_values = next_state_values.detach()

        # calculate the target Q value
        expected_state_action_values = (next_state_values * self.gamma) + reward
        losses = self.losses.parse_losses(state_action_values, expected_state_action_values.to(torch.float32))
        # sum up all losses
        losses.update({'loss': sum([losses[k] for k in losses.keys() if 'loss' in k.lower()])})
        return losses

    def soft_update(self):
        # Soft update of target network
        if self.trainer.global_step % self.update_interval == 0:
            self.target_agent.load_state_dict(self.main_agent.state_dict())

    def forward_test(self, x, label=None):
        raise NotImplementedError('forward_test is not implemented for BaseAgent')




