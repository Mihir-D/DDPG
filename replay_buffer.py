from collections import deque
import numpy as np
import random

class ReplayBuffer:
	def __init__(self, size):
		# self.params = load_params(???)
		self.buffer_size = size
		self.buffer = deque()
		self.count = 0

	def add(self, state, action, reward, state_, done):
		experience = (state,action,reward,state_,done)
		if self.count < self.buffer_size:
			self.count = self.count + 1
			self.buffer.append(experience)
		else:
			self.buffer.popleft()
			self.buffer.append(experience)

	def sampleBatch(self, batch_size):
		batch = []
		return random.sample(self.buffer, batch_size)

	def clear(self):
		self.count = 0
		self.buffer.clear()


