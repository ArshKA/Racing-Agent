import json
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from helpers import process_batch, extract_features
from models import NPModel

NUM_AGENTS = 30
NUM_TRIALS = 3
AGENT_OFFSPRING_LIST = [10, 5, 4, 3, 2, 1]
TOTAL_GENERATIONS = 150

reward_list = []
all_agents = []
for i in range(NUM_AGENTS):
    all_agents.append(NPModel([[np.random.random((16, 32)), np.random.random()],
                               [np.random.random((8, 16)), np.random.random()],
                               [np.random.random((8, 8)), np.random.random()],
                               [np.random.random((3, 8)), np.random.random()]]))

# create a vectorized environment with NUM_AGENTS copies of LunarLander-v2
# envs = gym.vector.make("LunarLander-v2", num_envs=NUM_AGENTS)

last_obs = deque(maxlen=3)

envs = gym.vector.SyncVectorEnv([lambda: gym.make("CarRacing-v2")]*NUM_AGENTS)

for generation in range(TOTAL_GENERATIONS):
    start = time.time()
    # reset the vectorized environment and get a batch of observations
    observations, infos = envs.reset()
    for i in range(3):
        observations, r, dones, timeouts, infos = envs.step(np.zeros((NUM_AGENTS, 3)))
        observations = process_batch(observations)
        last_obs.append(observations)
    # initialize the rewards for all agents to zero
    rewards = np.zeros(NUM_AGENTS)
    # loop until all agents are done
    step = 0
    while step < 500:
        # get a batch of actions from all agents
        observations = extract_features(np.transpose(last_obs, (1, 2, 3, 0)))
        actions = np.array([agent.forward(observation) for agent, observation in zip(all_agents, observations)])
        # step the vectorized environment and get a batch of next observations, rewards, dones and infos
        observations, r, dones, timeouts, infos = envs.step(actions)
        observations = process_batch(observations)
        # update the rewards for all agents
        rewards += r
        step += 1
        last_obs.append(observations)

    # assign rewards to the agents
    for agent, reward in zip(all_agents, rewards):
        agent.reward = reward  # divide by NUM_TRIALS to get the average reward per trial
    all_agents.sort(key=lambda a: a.reward, reverse=True)
    reward_list.append(all_agents[0].reward)

    # save the weights of the best agent
    if (generation+1) % 10 == 0:
        with open('1DWeights.json', 'w') as f:
            print("Saving weights")
            json.dump([[x[0].tolist(), x[1]] for x in all_agents[0].weights], f)

    print("Generation:", generation+1, "Reward:", all_agents[0].reward, "Elapsed Time:", time.time()-start)
    all_agents = all_agents[:5]
    for rank, num in enumerate(AGENT_OFFSPRING_LIST):
        all_agents.extend(all_agents[rank].generate_offspring(num, scale=4))

envs.close()
plt.plot(reward_list)
plt.show()
