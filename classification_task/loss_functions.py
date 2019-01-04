# -*-coding: utf-8 -*-

import os

import torch
import torch.nn as nn


import math


# Loss function
class ClassificationLoss(nn.Module):
	"""docstring for SegmentationLoss"""
	
	def __init__(self, average=True):
		super(ClassificationLoss, self).__init__()
		self.loss_func = nn.CrossEntropyLoss(size_average=average)
	
	def forward(self, output, y_target):
		loss = self.loss_func(output, y_target)
		
		if math.isnan(loss.item()):
			print('loss = ', loss.item())
			print('output = ', output)
			print('y_target = ', y_target)
		
		return loss