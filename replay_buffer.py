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
		if self.count < self.buffer_size:
			self.count = self.count + 1
			self.buffer.append((state,action,reward,state_,done))
		else:
			self.buffer.popleft()
			self.buffer.append((state,action,reward,state_,done))

	def sampleBatch(self, batch_size):
		batch = []
		if self.count < batch_size:
			return self.buffer
		else:
			indices = random.sample(range(0, self.count), batch_size)
			for index in indices:
				batch.append(self.buffer[index])
		return batch

	def clear(self):
		self.count = 0
		self.buffer.clear()


