import random
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
from torch import nn
from collections import deque # this python module implements exactly what we need for the replay memeory
import glob
import io
import base64
import os
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('runs/task_1')

# display = Display(visible=0, size=(1400, 900))
# display.start()


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity) # Define a queue with maxlen "capacity"

    def push(self, state, action, next_state, reward):
        self.memory.append( (state, action, next_state, reward) ) # Add the tuple (state, action, next_state, reward) to the queue

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self)) # Get all the samples if the requested batch_size is higher than the number of sample currently in the memory
        pick = random.sample(self.memory, batch_size)
        return list(zip(*pick))

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


def choose_action_epsilon_greedy(device, net, state, epsilon):
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32).to(device) # Convert the state to tensor
        net_out = net(state)

    best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    if random.random() < epsilon:
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        action = random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action
    return action, net_out

def choose_action_softmax(device, net, state, temperature):
    if temperature < 0:
        raise Exception('The temperature value must be greater than or equal to 0 ')
    # If the temperature is 0, just select the best action using the eps-greedy policy with epsilon = 0
    if temperature == 0:
        return choose_action_epsilon_greedy(device, net, state, 0)
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        net_out = net(state)

    # Apply softmax with temp
    temperature = max(temperature, 1e-8) # set a minimum to the temperature for numerical stability
    softmax_out = nn.functional.softmax(net_out / temperature, dim=0).cpu().numpy()

    all_possible_actions = np.arange(0, softmax_out.shape[-1])
    action = np.random.choice(all_possible_actions, p=softmax_out) # this samples a random element from "all_possible_actions" with the probability distribution p (softmax_out in this case)
    return action, net_out


def update_step(device, policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
    # Sample the data from the replay memory
    batch = replay_mem.sample(batch_size)
    batch_size = len(batch[0])
    #states = torch.tensor(batch[:, 0]).float().to(device)
    #actions = batch[:, 1].int().to(device)
    #rewards = batch[:, 3].float().to(device)
    # Create tensors for each element of the batch
    #old_states      = torch.tensor([s[0] for s in batch], dtype=torch.float32).to(device)
    #old_actions     = torch.tensor([s[1] for s in batch], dtype=torch.int64).to(device)
    #old_rewards     = torch.tensor([s[3] for s in batch], dtype=torch.float32).to(device)
    states      = torch.tensor(batch[0], dtype=torch.float32).to(device)
    actions     = torch.unsqueeze(torch.tensor(batch[1]).to(device), 1)
    rewards     = torch.tensor(batch[3], dtype=torch.float32).to(device)
    non_final_next_states = torch.tensor(
        [el for el in batch[2] if el is not None],
        dtype=torch.float32
    ).to(device)
    non_final_mask = torch.tensor(
        [el is not None for el in batch[2]],
        dtype=torch.bool
    ).to(device)
    #old_non_final = torch.tensor([s[2] for s in batch if s[2] is not None], dtype=torch.float32).to(device) # the next state can be None if the game has ended
    #ipdb.set_trace()
    # Compute all the Q values (forward pass)
    policy_net.train()
    q_values = policy_net(states)
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions)

    # Compute the value function of the next states using the 
    # target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
      target_net.eval()
      q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = torch.zeros(batch_size).to(device)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0]
    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    # Compute the Huber loss
    loss = loss_fn(state_action_values, torch.unsqueeze(expected_state_action_values, 1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward() # Apply gradient clipping (clip all the gradients greater than 2 for training stability)
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()

def train(
    device,
    policy_net,
    target_net,
    replay_mem,
):
    initial_value = 5
    num_iterations = 1000

    # We compute the exponential decay in such a way the shape of 
    #the exploration profile does not depend on the number of iterations
    exp_decay = np.exp(-np.log(initial_value) / num_iterations * 6)
    exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]

    lr = 1e-2   # Optimizer learning rate
    #optimizer = torch.optim.SGD(policy_net.parameters(), lr=lr) # The optimizer will update ONLY the parameters of the policy network
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    gamma = 0.97   # gamma parameter for the long term reward
    target_net_update_steps = 10   # Number of episodes to wait before updating the target network
    end_training_threshold = 500*(target_net_update_steps - 1) + 490
    batch_size = 128   # Number of samples to take from the replay memory for each update
    bad_state_penalty = 0   # Penalty to the reward when we are in a bad state (in this case when the pole falls down) 
    min_samples_for_training = 1000   # Minimum samples in the replay memory to enable the training

    # Initialize the Gym environment
    env = gym.make('CartPole-v1')
    env.seed(0) # Set a random seed for the environment (reproducible results)

    scores = 0
    for episode_num, tau in enumerate(exploration_profile):

        # Reset the environment and get the initial state
        state = env.reset()
        # Reset the score. The final score will be the total amount of steps before the pole falls
        score = 0
        done = False

        # Go on until the pole falls off
        while not done:

          # Choose the action following the policy
          action, _ = choose_action_softmax(device, policy_net, state, temperature=tau)

          # Apply the action and get the next state, the reward and a flag "done" 
          #that is True if the game is ended
          next_state, reward, done, info = env.step(action)
          next_state = next_state
          # We apply a (linear) penalty when the cart is far from center
          pos_weight = 1
          reward = reward - pos_weight * np.abs(state[0])

          # Update the final score (+1 for each step)
          score += 1

          # Apply penalty for bad state
          if done: # if the pole has fallen down 
              reward += bad_state_penalty
              next_state = None

          replay_mem.push(state, action, next_state, reward)
          # Update the network
          if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
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

        writer.add_scalar('score', score, global_step=episode_num+1)#
        scores += score
        # Update the target network every target_net_update_steps episodes
        if episode_num % target_net_update_steps == 0:
            print('Updating target network...')
            target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network
            if scores >= end_training_threshold:
                print("Reached end_training_threshold, done training.")
                break
            scores = 0
        # Print the final score
        print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}") # Print the final score
    env.close()


def test(policy_net, device):
    # Initialize the Gym environment
    env = gym.make('CartPole-v1')
    env.seed(1) # Set a random seed for the environment (reproducible results)
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
          action, q_values = choose_action_softmax(device, policy_net, state, temperature=0)
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
    ### Define exploration profile
    env = gym.make('CartPole-v1')
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
    )
    test(
        policy_net,
        device
    )

if __name__ == '__main__':
    main()
