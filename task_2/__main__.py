import random
import torch
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
from torch import nn
from collections import deque
import os

from .net import Net as DQN
from .img_manager import ImgManager

from pyvirtualdisplay import Display
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/task_2')

display = Display(visible=0, size=(1400, 900))
display.start()


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(
            maxlen=capacity)  # Define a queue with maxlen "capacity"

    def push(self, state, action, next_state, reward):
        self.memory.append(
            (state, action, next_state, reward)
        )  # Add the tuple (state, action, next_state, reward) to the queue

    def sample(self, batch_size):
        batch_size = min(
            batch_size, len(self)
        )  # Get all the samples if the requested batch_size is higher than the number of sample currently in the memory
        pick = random.sample(
            self.memory, batch_size)  # Randomly select "batch_size" samples
        return list(zip(*pick))

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self):

        torch.cuda.empty_cache()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.env = gym.make('CartPole-v1')
        self.img_manager = ImgManager(self.env)
        ### Define exploration profile
        initial_value = 5.0
        num_iterations = 2000
        exp_decay = np.exp(
            -np.log(initial_value) / num_iterations * 6
        )  # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
        self.exploration_profile = [
            initial_value * (exp_decay**i) for i in range(num_iterations)
        ] if initial_value > 0 else np.zeros(num_iterations)
        self.exploration_profile = np.concatenate(
            (self.exploration_profile, np.zeros(10000) + 0.001))
        state_space_dim = self.env.observation_space.shape[0]
        action_space_dim = self.env.action_space.n

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        replay_memory_capacity = 10000  # Replay memory capacity
        self.replay_mem = ReplayMemory(replay_memory_capacity)
        self.policy_net = DQN(state_space_dim,
                              action_space_dim).to(self.device)
        self.target_net = DQN(state_space_dim,
                              action_space_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def choose_action_epsilon_greedy(self, net, state, epsilon):
        if epsilon > 1 or epsilon < 0:
            raise Exception('The epsilon value must be between 0 and 1')
        # Evaluate the network output from the current state
        with torch.no_grad():
            net.eval()
            #state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32).to(self.device),0) # Convert the state to tensor

            net_out = torch.squeeze(net(state.to(self.device)))
            #net_out = torch.squeeze(net(state))

        # Get the best action (argmax of the network output)
        best_action = int(net_out.argmax())
        # Get the number of possible actions
        action_space_dim = net_out.shape[-1]

        # Select a non optimal action with probability epsilon, otherwise choose the best action
        if random.random() < epsilon:
            # List of non-optimal actions
            non_optimal_actions = [
                a for a in range(action_space_dim) if a != best_action
            ]
            # Select randomly
            action = random.choice(non_optimal_actions)
        else:
            # Select best action
            action = best_action
        return action, net_out

    def choose_action_softmax(self, net, state, temperature):
        if temperature < 0:
            raise Exception(
                'The temperature value must be greater than or equal to 0 ')
        # If the temperature is 0, just select the best action using the eps-greedy policy with epsilon = 0
        if temperature == 0:
            return self.choose_action_epsilon_greedy(net, state, 0)
        # Evaluate the network output from the current state
        with torch.no_grad():
            net.eval()
            #state = torch.tensor(state, dtype=torch.float32).to(device)
            #net_out = torch.squeeze(net(torch.unsqueeze(state.to(device),0)))

            net_out = torch.squeeze(net(state.to(self.device)))
        # Apply softmax with temp
        temperature = max(
            temperature,
            1e-8)  # set a minimum to the temperature for numerical stability
        softmax_out = nn.functional.softmax(net_out / temperature,
                                            dim=0).cpu().numpy()

        # Sample the action using softmax output as mass pdf
        all_possible_actions = np.arange(0, softmax_out.shape[-1])
        action = np.random.choice(
            all_possible_actions, p=softmax_out
        )  # this samples a random element from "all_possible_actions" with the probability distribution p (softmax_out in this case)

        return action, net_out

    def update_step(self, gamma, optimizer, loss_fn, batch_size):
        # Samplethe data from the replay memory
        batch = self.replay_mem.sample(batch_size)
        batch_size = len(batch[0])
        states = torch.cat(batch[0]).float()
        actions = torch.tensor(batch[1], dtype=torch.int64)
        rewards = torch.tensor(batch[3]).float()
        non_final_next_states = torch.cat(
            [s for s in batch[2] if s is not None]).to(self.device)
        non_final_mask = torch.tensor([s is not None for s in batch[2]],
                                      dtype=torch.bool)

        # Compute all the Q values (forward pass)
        self.policy_net.train()
        #q_values = batch_apply(policy_net, states, self.device)
        q_values = self.policy_net(states.to(self.device))
        # Select the proper Q value for the corresponding action taken Q(s_t, a)
        state_action_values = q_values.gather(
            1,
            actions.unsqueeze(1).to(self.device))

        # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
        with torch.no_grad():
            self.target_net.eval()
            q_values_target = self.target_net(
                non_final_next_states.to(self.device))
        next_state_max_q_values = torch.zeros(batch_size).to(self.device)
        next_state_max_q_values[non_final_mask.to(
            self.device)] = q_values_target.max(dim=1)[0]
        next_state_max_q_values = next_state_max_q_values
        # Compute the expected Q values
        expected_state_action_values = rewards.to(
            self.device) + next_state_max_q_values * gamma
        expected_state_action_values = expected_state_action_values.unsqueeze(
            1)  # Set the required tensor shape
        # Compute the Huber loss
        loss = loss_fn(
            torch.squeeze(state_action_values),
            torch.squeeze(expected_state_action_values).to(self.device))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 2)
        optimizer.step()

    def train(self):
        # import pdb; pdb.set_trace()
        lr = 1e-5  # Optimizer learning rate
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()
        gamma = 0.97  # gamma parameter for the long term reward
        target_net_update_steps = 10
        batch_size = 20  # 64# 128
        bad_state_penalty = 0
        min_samples_for_training = 1000
        scores = 0
        for episode_num, tau in enumerate(self.exploration_profile):

            self.env.reset()
            state = self.img_manager.get_image()
            score = 0
            done = False

            while not done:
                action, _ = self.choose_action_softmax(self.policy_net, state,
                                                       tau)
                _, reward, done, info = self.env.step(action)
                next_state = self.img_manager.get_image()

                score += 1

                if done:  # if the pole has fallen down
                    reward += bad_state_penalty
                    next_state = None

                self.replay_mem.push(state, action, next_state, reward)

                if len(self.replay_mem) > min_samples_for_training:
                    for _ in range(1):
                        self.update_step(gamma, optimizer, loss_fn, batch_size)
                state = next_state
            scores += score
            if episode_num % target_net_update_steps == 0:
                print('Updating target network...')
                self.target_net.load_state_dict(self.policy_net.state_dict())
                if scores >= 450 * target_net_update_steps:
                    print("Reached quality threshold")
                    break
                scores = 0
            writer.add_scalar('score', score, global_step=episode_num)
            print(
                f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - T: {tau}"
            )
        self.env.close()
        self.save()

    def save(self):
        torch.save(self.policy_net.state_dict(), 'policy_net.pt')

    def test(self):
        for num_episode in range(10):
            self.env.reset()
            state = self.img_manager.get_image()
            score = 0
            done = False
            while not done:
                action, q_values = self.choose_action_softmax(self.policy_net,
                                                              state,
                                                              temperature=0)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.img_manager.get_image()
                score += reward
                state = next_state
            print(f"EPISODE {num_episode + 1} - FINAL SCORE: {score}")
        self.env.close()


def main():
    print('Initializing agent')
    agent = Agent()
    print('Starting to train')
    agent.train()
    print('Starting test')
    agent.test()


if __name__ == '__main__':
    main()
