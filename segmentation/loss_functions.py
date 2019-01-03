# -*-coding: utf-8 -*-


import torch.nn as nn
import math


# Loss function
class SegmentationLoss(nn.Module):
	""" Computes the loss for each pixel """
	
	def __init__(self, average=True):
		super(SegmentationLoss, self).__init__()
		self.loss_func = nn.CrossEntropyLoss(size_average=average)
	
	def forward(self, output, y_target):
		assert len(
			output.shape) == 4, 'output must be four dimensional where dim: (#batch-size, #num_of_classes, #height, #width)'
		
		# Each pixel is classified thus output #of channels which equals #of classes thus data is permuted in following way
		output = output.permute(0, 2, 3, 1)  # (#batch-size, #num_of_classes, #height, #width) = > (#batch-size, #height, #width, #num_of_classes)
		output = output.view(-1, output.shape[-1])
		
		y_target = y_target.view(-1)
		loss = self.loss_func(output, y_target)
		
		if math.isnan(loss.item()):
			print('loss = ', loss.item())
			print('output = ', output)
			print('y_target = ', y_target)
		
		return loss