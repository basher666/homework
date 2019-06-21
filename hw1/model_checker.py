import gym
import torch
import numpy as np
from models import policy
import pickle as pkl

obs_stats = pkl.load(open("obs_stats.pkl",'rb'))

policy_fn = policy(376,100,50,17)
policy_fn.load_state_dict(torch.load("humanoid_policy.model"))

env = gym.make("Humanoid-v2")
max_steps = env.spec.timestep_limit
totalr = 0
steps = 0
done = False
obs = env.reset()

while True:
	obs = (obs - obs_stats['mean'])/(obs_stats['std'] + 1e-6)
	action = np.asarray(policy_fn(torch.from_numpy(np.asarray(obs,dtype=np.float32))).data)
	# action = env.action_space.sample()
	obs, r, done, _ = env.step(action)
	totalr += r
	steps += 1
	env.render()
	if steps >= max_steps:
		break

print("total reward :",totalr)