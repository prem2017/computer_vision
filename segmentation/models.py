# -*-coding: utf-8 -*-



import torch.nn as nn

# Convolutional Model
class ConvNet(nn.Module):
	"""ConvNet for classification"""
	
	def __init__(self, num_classes=2, in_channels=3):
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
		# self.up_sample = nn.Upsample(scale_factor=2, mode='area')  # This is not learnable upsampling but will do https://pytorch.org/docs/stable/_modules/torch/nn/modules/upsampling.html
		self.layer3 = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=5, stride=1, padding=2)
	# output size WxHxnum_classes i.e. 2 for binary classification of pixel
	
	def forward(self, x):
		out = self.layer1(x)
		out = self.down_sample(out)
		out = self.layer2(out)
		out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
		out = self.layer3(out)
		return out
	
	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features