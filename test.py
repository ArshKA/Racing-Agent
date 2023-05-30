from models import NPModel
import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
from collections import deque
from PIL import Image
import imageio

from helpers import process_img, process_batch, extract_features


def create_gif(frames, filename, fps=20, loop=0):
  # Convert the frames to PIL images.
  pil_images = [Image.fromarray(frame) for frame in frames]

  # Save the GIF.
  imageio.mimsave(filename, pil_images, duration=len(pil_images)/60, loop=loop)

with open('pretrained_weights.json', 'r') as f:
    weights = json.load(f)

weights = [[np.array(x[0]), x[1]] for x in weights]

agent = NPModel(weights)

env = gym.make("CarRacing-v2", render_mode="rgb_array")
agent.reward = 0
observation, info = env.reset(seed=86)
step = 0
last_obs = deque(maxlen=3)

frames = []

for i in range(3):
    frames.append(env.render())
    observation, r, dones, timeouts, infos = env.step([0, 0, 0])
    observation = process_img(observation)
    last_obs.append(observation)

while step < 2000:
    if step%100 == 0:
        print(step)
    frames.append(env.render())
    observation = extract_features(np.transpose(last_obs, (1, 2, 3, 0)))[0]
    action = agent.forward(observation)
    observation, reward, complete, timeout, info = env.step(action)
    observation = process_img(observation)
    agent.reward += reward
    step += 1
    last_obs.append(observation)

print(agent.reward)


create_gif(frames, filename='test4.gif')
env.close()
