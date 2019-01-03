# -*- coding: utf-8 -*-


import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as torch_transformer
import torch.nn as nn




import numpy as np
import skimage.io as io
import skimage.transform as sk_transformer

import matplotlib.pyplot as plt
import pandas as pd



################ Series of Transformers  ################
# Rescale/Resize the with the given dim
class RescaleImage(object):
	""" RescaleImage the given for the given dimension

	Args:
		output_size: tuple => (height, width)
	"""
	
	def __init__(self, output_size, num_classes=2):
		self.output_size = output_size
		self.num_classes = num_classes
	
	def __call__(self, sample):
		image_ndarray, gt_ndarray = sample['rgb'], sample['gt']
		new_h, new_w = self.output_size
		
		new_img = sk_transformer.resize(image_ndarray, (new_h, new_w))
		# original_size = image_ndarray.shape
		# if original_size > self.output_size:
		# 	new_img = sk_transformer.resize(image_ndarray, (new_h, new_w))
		# else:
		# 	{'rgb': image_ndarray, 'gt': gt_ndarray}
		
		new_gt = gt_ndarray
		
		# TODO: resizing not handled for more than two classes
		if self.num_classes == 2:
			if new_gt is not None:
				new_gt = sk_transformer.resize(new_gt, (new_h, new_w))
				new_gt[new_gt>=0.1] = 255
				new_gt[new_gt<0.1] = 0
		else:
			# warn('Resizing is not available for more than two classes.')
			Exception('Invalid resizing for more than two class segmentation problem')
		
		return {'rgb': new_img, 'gt': new_gt}


# Randomly crop the image
class RandomCropImage(object):
	""" Crops image randomly for given size

	Args:
		output_size (tuple => (height, width)): Desired output size with height and width dimension respectively.
		Recall that that image is read with height and width dimension.
	"""
	
	def __init__(self, output_size):
		self.output_size = output_size
	
	def __call__(self, sample):
		image_ndarray, gt_ndarray = sample['rgb'], sample['gt']
		# assert (image_ndarray.shape<=3), 'First dimension must be width for transformation'
		height, width = image_ndarray.shape[:2]
		
		new_h, new_w = self.output_size
		
		top_ss = np.random.randint(0, (
					height - new_h) + 1)  # +1 for handling the case if original image size and output size matches
		left_ss = np.random.randint(0, (width - new_w) + 1)
		
		new_img = image_ndarray[top_ss: top_ss + new_h, left_ss: left_ss + new_w]
		
		new_gt = gt_ndarray
		if gt_ndarray is not None:
			new_gt = new_gt[top_ss: top_ss + new_h, left_ss: left_ss + new_w]
			
		return {'rgb': new_img, 'gt': new_gt}

# Tensories the ndarray and also sample them between 0-1
class ToTensor(object):
	""" Convert ndarray to image to Tensor """
	
	def __init__(self):
		self.to_tensor = torch_transformer.ToTensor()
	
	def __call__(self, sample):
		# Swap color axis because
		# numpy image: H x W x C
		# torch image: C x H x W
		# image_ndarray = image_ndarray.transpose(2, 0, 1)
		
		# Use the Torch method to transform as it transforms the data between 0-1
		"""
		transforms.ToTensor?
		Init signature: transforms.ToTensor()
		Docstring:
		Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

		Converts a PIL Image or numpy.ndarray (H x W x C) in the range
		[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
		File:           ~/anaconda/envs/torch_04/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/transforms/transforms.py
		Type:           type
		"""
		image_ndarray, gt_ndarray = sample['rgb'], sample['gt']
		assert isinstance(image_ndarray, np.ndarray), '[Assertion Error]: image must be of type numpy.ndarray'
		image_tensor = self.to_tensor(image_ndarray)
		
		# TODO: handle differently if it is labels instead of pixel output
		# TODO: Also if for multiclass handle it differently. Create a map {RGB => label} & {label => RGB}
		new_gt_tensor = gt_ndarray
		if new_gt_tensor is not None:
			if new_gt_tensor.ndim == 3:
				new_gt_tensor = new_gt_tensor.transpose(2, 0, 1)
			new_gt_tensor = torch.tensor(new_gt_tensor) # self.to_tensor(new_gt_tensor)
			
		return {'rgb': image_tensor, 'gt': new_gt_tensor}


class NormalizeImageData(object):
	""" Normalize the image data per channel for quicker and better training """
	
	def __call__(self, sample):
		img_tensor, gt_tensor = sample['rgb'], sample['gt']
		"""
		:param img_tensor: A single image tensor of shape (C x H x W)
		:param gt_tensor: None or tensor of shape (C x H x W) {C = 1 or 3}
		:return: Return normalized sample
		"""
		
		assert img_tensor.dim() == 3, '[Assertion Error]: Normalization is performed for each channel so input must be only one image data only'
		img_tensor = img_tensor.contiguous()
		mean_list = img_tensor.view(img_tensor.shape[0], -1).mean(dim=1)
		std_list = img_tensor.view(img_tensor.shape[0], -1).std(dim=1)
		
		normalizer = torch_transformer.Normalize(mean=mean_list, std=std_list)
		
		new_gt = gt_tensor
		# ground truth is actually labels so we need to convert to just one channel
		if new_gt is not None:
			new_gt = convert_gt_to_label(new_gt) # Only one channel where each entry is either label or mapped to label
			print('[new_gt] ', new_gt.shape)
		return  {'rgb': normalizer(img_tensor), 'gt': new_gt}


# TODO: This is fine for only binary class. For more classes eithe it should already be label (then nothing needs to be done) or else we should have map of {summed_value => label}
def convert_gt_to_label(gt_tensor):
	
	# assert (gt_tensor.shape[0] == (1, 3))
	if gt_tensor.dim() == 3:
		gt_tensor = gt_tensor.sum(dim=0)
	
	gt_tensor[gt_tensor < 0.01] = 0
	gt_tensor[gt_tensor > 0.01] = 1
		
	return gt_tensor