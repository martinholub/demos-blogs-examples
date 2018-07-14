import gym
import gym.spaces
import matplotlib
# matplotlib.use("Agg") # use noninteractive backend
import matplotlib.pyplot as plt

gym.logger.set_level(40)
gym.__version__

import autograd.numpy as np
from autograd import grad, elementwise_grad
import random

### Model ---------------------------------------------------------------------
# Linear approximation function to expected returns
def approx(weights, observation, action):
    return np.dot(observation, weights)[action]

def policy(env, weights, observation, epsilon):
    actions = [0, 1]
    if np.random.rand() < epsilon:
        return random.choice(actions)

    qs = []
    for action in actions:
        qs.append(approx(weights, observation, action))
    return np.argmax(qs)

dapprox = grad(approx)
discount = 1.0 # Discount rate
epsilon = 0.2 # Exploration rate
alpha = 0.1 # Step size for gradient descent
w = np.zeros((4,2)) # Initalize weigths
num_episodes = 1000 # Number of games for the agent to play
max_steps = 200

### Plotting --------------------------------------------------------------------
import os
import tempfile
import subprocess
anim_path = "./monitor"
if not os.path.isdir(anim_path): os.makedirs(anim_path)

def save_frames(frames, anim_path):
    temp_dir = tempfile.mkdtemp(dir = anim_path)
    for i, frame in enumerate(frames[-200:]):
        plt.imsave(*("{}/{}.png".format(temp_dir, i), frame),
                        **{"vmin":0, "vmax":255})
    subprocess.run("ffmpeg -y -f image2 -i {0}/%d.png {0}/video.avi".format(temp_dir),
                        shell = True)
    subprocess.run("ffmpeg -y -i {}/video.avi -r 9 {}/anim.gif".format(temp_dir, anim_path),
                        shell = True)

    os.rmdir(temp_dir)


### Training ------------------------------------------------------------------
from collections import deque
from gym import wrappers

env = gym.make('CartPole-v0')
episode_rewards = []
for ep in range(num_episodes):
    state = env.reset()
    rewards = []
    frames = deque(maxlen = 500)
    for _ in range(max_steps):
        # Take smart action based on defined policy
        action = policy(env, w, state, epsilon)

        q_hat = approx(w, state, action)
        q_hat_grad = dapprox(w, state, action)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        # Render into buffer.
        visframe = env.render(mode = 'rgb_array')
        frames.append(visframe)
        if done:
            w += alpha*(reward - q_hat) * q_hat_grad
            break
        else:
            # Update weights to maximize for reward
            next_action = policy(env, w, next_state, epsilon)
            q_hat_next = approx(w, next_state, next_action)
            w += alpha*(reward - discount*q_hat_next)*q_hat_grad
            state = next_state
    # Reguralizer
    # as we learn more about the game, become more certain in making decision
    if ep == 100:
        epsilon /= 2

    episode_rewards.append(np.sum(rewards))
    mean_reward=np.mean(episode_rewards[max(ep-100, 0):ep+1])

    # Report on progress - did we solve the task already?
    if mean_reward >= 195.0 and ep >= 100:
        print("Episodes before solve {}".format(ep-100+1))
        save_frames(frames, anim_path)
        break
    if ((ep % 100) == 0) and ep > 0:
        print("Episode {}/{} finished. Mean reward over last 100 episodes: {:.2f}"\
              .format(ep, num_episodes, (mean_reward)))
env.close()
