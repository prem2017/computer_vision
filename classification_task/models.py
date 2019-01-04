# -*-coding: utf-8 -*-

import os

import torch
import torch.nn as nn




# Model
class ConvNet(nn.Module):
	"""ConvNet for classification"""
	def __init__(self, num_classes=2, in_channels=3, width=256, height=256, dropout=0.5):
		super(ConvNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
			nn.ReLU())
		self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
			nn.ReLU())
		self.layer3 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=5, stride=1, padding=2)
		# output size WxHx2
		self.fc1 = nn.Sequential(
			nn.Linear(width * height * 2, 128),
			nn.Dropout(p=dropout),
			nn.ReLU())
		self.fc2 = nn.Linear(128, num_classes)
	
	
	
	def forward(self, x):
		out = self.layer1(x)
		out = self.down_sample(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = out.view(-1, self.num_flat_features(out))
		out = self.fc1(out)
		out = self.fc2(out)
		return out
	
	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features