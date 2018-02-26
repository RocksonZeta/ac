import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import gym

env = gym.make("CartPole-v0")

N_S = env.observation_space.shape[0]
N_A = env.action_space.n
GAMMA = 0.9
STEP =10

print("env state dim:",N_S , ",action dim:",N_A)

def from_numpy(np_array,dtype=np.float32):
	if np_array.dtype!=dtype :
		np_array = np_array.astype(dtype)
	return Variable(torch.from_numpy(np_array))

class ACNet(nn.Module):
	'''
	V: s -> r(scalar)
	Pi: s -> distribution of action
	'''
	def __init__(self,s_dim ,a_dim):
		super(ACNet,self).__init__()
		self.s_dim = s_dim 
		self.a_dim = a_dim 
		self.v1 = nn.Linear(s_dim,100)
		self.v2 = nn.Linear(100,1)
		self.pi1 = nn.Linear(s_dim , 100)
		self.pi2 = nn.Linear(100,a_dim)
		self.dist_cate = torch.distributions.Categorical

	def forward(self,state):
		pi = F.relu(self.pi1(state))
		actions = self.pi2(pi)

		v = F.relu(self.v1(state))
		values = self.v2(v)
		return actions , values

	def loss_func(self,s,a,v_targets):
		self.train()
		#logit and values are logical, should close to reality
		logits,values = self.forward(s)
		td = v_targets - values # logical reality difference
		c_loss = td.pow(2)
		# actor's aciton distribution a(a|s,theta)
		action_pb = F.softmax(logits , dim=1)
		pi_dist = self.dist_cate(action_pb)
		a_loss = -pi_dist.log_prob(a) * td.detach()

		# we should minimize critic loss first and then minimize actor loss
		total_loss = (c_loss+a_loss).mean()
		return total_loss

	def choose_action(self , state):
		'''
		state : single state
		'''
		self.eval()
		states = torch.unsqueeze(from_numpy(state),dim=0)
		actions,_ = self.forward(states)
		pb = F.softmax(actions , dim=1).data
		return self.dist_cate(pb).sample().numpy()[0]
	def update(self,opt , s_t,states,actions,rs,done):
		'''
		n-step learning,
		s_t: last state
		states : state t,t+step
		actions : action t ,t+step
		rs : rewards 
		done :last state is done?
		'''
		if done:
			R = 0
		else : 
			s_t = torch.unsqueeze(from_numpy(s_t[None,:]) , 0)
			R = self.forward(s_t)[-1].data.numpy()[0]
		v_target = []
		for r in rs[::-1]:
			R = r + GAMMA * R
			v_target.append(R)
		v_target.reverse()
		v_target = from_numpy(np.stack(v_target ,0))
		loss = self.loss_func(states , actions , v_target)
		opt.zero_grad()
		loss.backward()
		opt.step()
		# print("loss:" ,loss.data[0])
		
		
	@classmethod
	def training(cls , env):
		ac = ACNet(N_S , N_A)
		opt = torch.optim.Adam(ac.parameters())		
		for epi in range(2005):
			s = env.reset()
			ab ,sb,rb = [],[],[]
			total_r =0			
			for i in range(200):
				if epi> 2000:
					env.render()
				a = ac.choose_action(s)
				s_,r , done ,_ = env.step(a)
				if done : 
					r = -1
				ab.append(a)
				sb.append(s)
				rb.append(r)
				total_r += r
				if i % STEP == 0 and i>0 :
					ab = from_numpy(np.stack(ab) , dtype=np.int64)
					sb = from_numpy(np.stack(sb))
					# rb = from_numpy(np.stack(rb))
					ac.update(opt,s_ , sb,ab, rb ,done)
					ab ,sb,rb = [],[],[]
				s = s_
				if done :
					break
			if epi%20 ==0 :
				print("total reward:",total_r)
			# actions ,values = ac.forward(sb)
			
			
ACNet.training(env)