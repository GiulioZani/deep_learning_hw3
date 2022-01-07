import random
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from collections import deque # this python module implements exactly what we need for the replay memeory
import glob
import io
import base64
import os
#from IPython import display as ipythondisplay
#from pyvirtualdisplay import Display
#from gym.wrappers import Monitor
from torch.utils.tensorboard import SummaryWriter
import ipdb

writer = SummaryWriter('runs/lab_07')

#display = Display(visible=0, size=(1400, 900))
#display.start()


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity) # Define a queue with maxlen "capacity"

    def push(self, state, action, next_state, reward):
        self.memory.append( (state, action, next_state, reward) ) # Add the tuple (state, action, next_state, reward) to the queue

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self)) # Get all the samples if the requested batch_size is higher than the number of sample currently in the memory
        return random.sample(self.memory, batch_size) # Randomly select "batch_size" samples

    def __len__(self):
        return len(self.memory) # Return the number of samples currently stored in the memory


class DQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()

        self.linear = nn.Sequential(
                nn.Linear(state_space_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, action_space_dim)
                )

    def forward(self, x):
        return self.linear(x)


# ## Exploration Policy

# Starting from the estimated Q-values, we need to choose the proper action. This action may be the one expected to provide the highest long term reward (exploitation), or maybe we want to find a better policy by choosing a different action (exploration).
# The exploration policy controls this behavior, typically by varying a single parameter.
# Since our Q-values estimates are far from the true values at the beginning of the training, a high exploration is preferred in the initial phase.
# The steps are:
# `Current state -> Policy network -> Q-values -> Exploration Policy -> Action`

# ### Epsilon-greedy policy
# With an epsilon-greedy policy we choose a **non optimal** action with probability epsilon, otherwise choose the best action (the one corresponding to the highest Q-value).
def choose_action_epsilon_greedy(device, net, state, epsilon):
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32).to(device) # Convert the state to tensor
        net_out = net(state)

    # Get the best action (argmax of the network output)
    best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if random.random() < epsilon:
        # List of non-optimal actions
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        # Select randomly
        action = random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action
    return action, net_out

# ### Softmax policy
# With a softmax policy we choose the action based on a distribution obtained applying a softmax (with temperature $\tau$) to the estimated Q-values. The highest the temperature, the more the distribution will converge to a random uniform distribution. At zero temperature, instead, the policy will always choose the action with the highest Q-value.
def choose_action_softmax(device, net, state, temperature):
    if temperature < 0:
        raise Exception('The temperature value must be greater than or equal to 0 ')
    # If the temperature is 0, just select the best action using the eps-greedy policy with epsilon = 0
    if temperature == 0:
        return choose_action_epsilon_greedy(net, state, 0)
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        net_out = net(state)

    # Apply softmax with temp
    temperature = max(temperature, 1e-8) # set a minimum to the temperature for numerical stability
    softmax_out = nn.functional.softmax(net_out / temperature, dim=0).cpu().numpy()

    # Sample the action using softmax output as mass pdf
    all_possible_actions = np.arange(0, softmax_out.shape[-1])
    action = np.random.choice(all_possible_actions, p=softmax_out) # this samples a random element from "all_possible_actions" with the probability distribution p (softmax_out in this case)

    return action, net_out


# ### Exploration profile

# Let's consider, for example, an exponentially decreasing exploration profile using a softmax policy.
# 
# $$
# \text{softmax_temperature}  = \text{initial_temperature} * \text{exp_decay}^i \qquad \text{for $i$ = 1, 2, ..., num_iterations } 
# $$
# 
# Alternatively, you can consider an epsilon greedy policy. In this case the exploration would be controlled by the epsilon parameter, for which you should consider a different initial value (max 1). 



# ## Update function
def update_step(device, policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
    # Sample the data from the replay memory
    batch = replay_mem.sample(batch_size)
    batch_size = len(batch)

    # Create tensors for each element of the batch
    states      = torch.tensor([s[0] for s in batch], dtype=torch.float32).to(device)
    actions     = torch.tensor([s[1] for s in batch], dtype=torch.int64).to(device)
    rewards     = torch.tensor([s[3] for s in batch], dtype=torch.float32).to(device)

    # Compute a mask of non-final states (all the elements where the next state is not None)
    non_final_next_states = torch.tensor([s[2] for s in batch if s[2] is not None], dtype=torch.float32).to(device) # the next state can be None if the game has ended
    non_final_mask = torch.tensor(
        [s[2] is not None for s in batch], dtype=torch.bool
    ).to(device)

    # Compute all the Q values (forward pass)
    policy_net.train()
    q_values = policy_net(states)
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1))

    # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
      target_net.eval()
      q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = torch.zeros(batch_size).to(device)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0]

    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1) # Set the required tensor shape

    # Compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping (clip all the gradients greater than 2 for training stability)
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()

def train(
    device,
    policy_net,
    target_net,
    replay_mem,
    exploration_profile,
):
    lr = 1e-2   # Optimizer learning rate
    optimizer = torch.optim.SGD(policy_net.parameters(), lr=lr) # The optimizer will 
    # update ONLY the parameters of the policy network
    #optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    gamma = 0.97   # gamma parameter for the long term reward
    target_net_update_steps = 10   # Number of episodes to wait before updating the 
    # target network
    batch_size = 128   # Number of samples to take from the replay memory for each update
    bad_state_penalty = 0   # Penalty to the reward when we are in a bad state (in this case when the pole falls down) 
    min_samples_for_training = 1000   # Minimum samples in the replay memory to enable the training

    # Initialize the Gym environment
    env = gym.make('CartPole-v1')
    env.seed(0) # Set a random seed for the environment (reproducible results)

    # This is for creating the output video in Colab, not required outside Colab

    for episode_num, tau in enumerate(tqdm(exploration_profile)):

        # Reset the environment and get the initial state
        state = env.reset()
        # Reset the score. The final score will be the total amount of steps before 
        # the pole falls
        score = 0
        done = False
        # Go on until the pole falls off
        while not done:
            # Choose the action following the policy
            action, _ = choose_action_softmax(device, policy_net, state, temperature=tau)
            # Apply the action and get the next state, the reward and a flag "done" that is 
            # True if the game is ended
            next_state, reward, done, info = env.step(action)
            # We apply a (linear) penalty when the cart is far from center
            pos_weight = 1
            reward = reward - pos_weight * np.abs(state[0])
            # Update the final score (+1 for each step)
            score += 1

            # Apply penalty for bad state
            if done: # if the pole has fallen down 
                reward += bad_state_penalty
                next_state = None
            # Update the replay memory
            replay_mem.push(state, action, next_state, reward)
            # Update the network
            if len(replay_mem) > min_samples_for_training: # we enable the training 
                # only if we have enough samples in the replay memory, otherwise the
                # training will use the same samples too often
                update_step(
                      device,
                      policy_net,
                      target_net,
                      replay_mem,
                      gamma,
                      optimizer,
                      loss_fn,
                      batch_size
                )
            # Set the current state for the next iteration
            state = next_state
            writer.add_scalar('score', score, global_step=episode_num + 1)
            # Update the target network every target_net_update_steps episodes
        if episode_num % target_net_update_steps == 0:
            print('Updating target network...')
            target_net.load_state_dict(policy_net.state_dict()) # This will copy
            # the weights of the policy network to the target network

        # Print the final score
        print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}")
        env.close()


def test(policy_net):
    # Initialize the Gym environment
    env = gym.make('CartPole-v1')
    env.seed(1) # Set a random seed for the environment (reproducible results)

    # This is for creating the output video in Colab, not required outside Colab
    env = wrap_env(env, video_callable=lambda episode_id: True) # Save a video every episode

    # Let's try for a total of 10 episodes
    for num_episode in range(10):
        # Reset the environment and get the initial state
        state = env.reset()
        # Reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        done = False
        # Go on until the pole falls off or the score reach 490
        while not done:
          # Choose the best action (temperature 0)
          action, q_values = choose_action_softmax(policy_net, state, temperature=0)
          # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
          next_state, reward, done, info = env.step(action)
          # Visually render the environment
          # Update the final score (+1 for each step)
          score += reward
          # Set the current state for the next iteration
          state = next_state
          # Check if the episode ended (the pole fell down)
        # Print the final score
        print(f"EPISODE {num_episode + 1} - FINAL SCORE: {score}")
    env.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CartPole-v1')
    ### Define exploration profile
    initial_value = 5
    num_iterations = 1000
    exp_decay = np.exp(-np.log(initial_value) / num_iterations * 6) # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
    exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]
    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    replay_memory_capacity = 10000   # Replay memory capacity
    replay_mem = ReplayMemory(replay_memory_capacity)
    policy_net = DQN(state_space_dim, action_space_dim).to(device)
    target_net = DQN(state_space_dim, action_space_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network
    train(
        device,
        policy_net,
        target_net,
        replay_mem,
        exploration_profile,
    )
    test(policy_net)

if __name__ == '__main__':
    main()
