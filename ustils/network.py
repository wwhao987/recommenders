#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/9 13:10
@Author  : 朱紫宇
@File    : network.py
@Software: PyCharm
"""

import torch.nn as nn

class Network(nn.Module):
	def __init__(self):
		super(Network,self).__init__()

		self.feature = nn.Sequential(
			nn.Linear(18,5),
			nn.ReLU(),
			nn.Linear(5,20),
			nn.ReLU(),
			nn.Linear(20,1)
		)


	def foward(self,x):
		return self.feature(x)
