import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import gym
import math
np.set_printoptions(linewidth=300)
env = gym.make("Pendulum-v0")
print(env.observation_space.shape)
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
GAMMA = 0.9
UPDATE_STEP = 20
EPISODES = 5000
L1_SIZE = 100

print("env state dim:" ,N_S ," action dim:", N_A ,"action bound", env.action_space.high, env.action_space.low)
def from_numpy(np_array,dtype=np.float32):
	if np_array.dtype!=dtype :
		np_array = np_array.astype(dtype)
	return Variable(torch.from_numpy(np_array))

def init_param(self,layers):
	for layer in layers:
		nn.init.normal(layer.weight ,0,0.1)
		nn.init.constant(layer.bias,0.)
class ACNet(nn.Module):
	def __init__(self, s_dim,a_dim):
		super(ACNet,self).__init__()
		self.a1 = nn.Linear(s_dim , L1_SIZE)
		self.mu = nn.Linear(L1_SIZE ,a_dim)
		self.sigma = nn.Linear(L1_SIZE , a_dim)

		self.v1 = nn.Linear(s_dim , L1_SIZE)
		self.v2 = nn.Linear(L1_SIZE , 1)

		self.init_param([self.a1,self.mu,self.sigma,self.v1,self.v2])
		# actors action distribution assume gaussian , we need mu and sigma, use nn to eval it
		self.actor_distribution = torch.distributions.Normal
		self.optim = torch.optim.Adam(self.parameters(),lr=0.0002)
		self.record = []
	def init_param(self,layers):
		for layer in layers:
			nn.init.normal(layer.weight ,0,0.1)
			nn.init.constant(layer.bias,0.)
		
	def forward(self,x):
		a1 = F.relu(self.a1(x))
		mu = 2* F.tanh(self.mu(a1))
		sigma = F.softplus(self.sigma(a1))
		
		v1 = F.relu(self.v1(x))
		values = self.v2(v1)
		return mu ,sigma , values

	def choose_action(self,state):
		self.eval()
		s = state[None , :]
		s = from_numpy(s)
		mu,sigma ,_ = self.forward(s)
		pi = self.actor_distribution(mu,sigma)
		return pi.sample().data.numpy()[0]

	def loss_func(self,s,a,v_targets):
		self.train()
		mu ,sigma ,values = self(s)
		# logic should close to reality
		td =v_targets - values #A
		critic_loss = td.pow(2)

		pi = self.actor_distribution(mu , sigma)
		entropy = 0.5 + 0.5 * math.log(2*math.pi) + torch.log(pi.std)
		#gradient policy, to max goal(cumulative rewards)
		actor_loss = -(pi.log_prob(a) * td.detach() + 0.005 * entropy)
		
		total_loss = (critic_loss+actor_loss).mean()
		return total_loss

	def update(self,states,actions,rewards,done,state_next):
		states = from_numpy(np.stack(states))
		actions = from_numpy(np.stack(actions))
		if done :
			value_t = 0 # last state game over , no value,
		else :
			value_t = self.forward(from_numpy(state_next[None,:]))[-1].data.numpy()[0] # using nn to eval value
		value_targets = []

		# Bellman equation : v(n) = E[r + gamma * v(n+1)] , E for trajectories
		for r in rewards[::-1]:
			value_t = r + GAMMA * value_t
			value_targets.append(value_t)
		value_targets.reverse()
		value_targets = from_numpy(np.stack(value_targets))
		loss = self.loss_func(states,actions,value_targets)
		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
		self.record.append(loss.data[0])
		

	@classmethod
	def training(cls,env):
		total_step = 1
		ac = ACNet(N_S , N_A)
		for epi in range(EPISODES):
			state = env.reset()
			states ,actions,rewards = [],[],[]
			episode_reward = 0
			for i in range(200):
				action = ac.choose_action(state)
				state_next,reward,done,_ = env.step(np.clip(action ,-2,2)) # action space is [-2,2]
				episode_reward += reward
				# show last 10
				if epi > EPISODES - 10 : 
					env.render()
				states.append(state)
				actions.append(action)
				rewards.append(reward)
				if total_step % UPDATE_STEP ==0 or done :
					ac.update(states ,actions, rewards , done,state_next)
					states ,actions,rewards = [],[],[]
				total_step +=1
				if done :
					break
				state = state_next
			print("episode",epi,"reward total:" , episode_reward)
		torch.save(ac,"ac_conti_net.pkl")
		ac.plot_record()
	def plot_record(self):
		plt.plot(self.record)
		plt.show()
	@staticmethod
	def restore():
		ac = torch.load("ac_conti_net.pkl")
		for epi in range(100):
			s = env.reset()
			rt = 0
			for i in range(200):
				action = ac.choose_action(s)
				s_,r,done,_=env.step(action)
				env.render()
				s =s_
				rt += r
				if done :
					break
			print("rewards:",rt)
ACNet.training(env)
# ACNet.restore()