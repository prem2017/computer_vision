# -*- coding:utf-8 -*-

import os

import torch
import torchvision.transforms as torch_transformer

import numpy as np
import skimage.transform as sk_transformer




################ Series of Transformers  ################
# Rescale/Resize the with the given dim
class RescaleImage(object):
	""" RescaleImage the given for the given dimension

	Args:
		output_size: tuple => (height, width)
	"""
	
	def __init__(self, output_size):
		self.output_size = output_size
	
	def __call__(self, image_ndarray):
		new_h, new_w = self.output_size
		
		new_img = sk_transformer.resize(image_ndarray, (new_h, new_w))
		return new_img


# Randomly crop the image
class RandomCropImage(object):
	""" Crops image randomly for given size

	Args:
		output_size (tuple => (height, width)): Desired output size with height and width dimension respectively.
		Recall that that image is read with height and width dimension.
	"""
	
	def __init__(self, output_size):
		self.output_size = output_size
	
	def __call__(self, image_ndarray):
		# assert (image_ndarray.shape<=3), 'First dimension must be width for transformation'
		height, width = image_ndarray.shape[:2]
		
		new_h, new_w = self.output_size
		
		top_ss = np.random.randint(0, (
					height - new_h) + 1)  # +1 for handling the case if original image size and output size matches
		left_ss = np.random.randint(0, (width - new_w) + 1)
		
		return image_ndarray[top_ss: top_ss + new_h, left_ss: left_ss + new_w]


class ToTensor(object):
	""" Convert ndarray to image to Tensor """
	
	def __init__(self):
		self.to_tensor = torch_transformer.ToTensor()
	
	def __call__(self, image_ndarray):
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
		assert isinstance(image_ndarray, np.ndarray), '[Assertion Error]: image must be of type numpy.ndarray'
		return self.to_tensor(image_ndarray)


class NormalizeImageData(object):
	""" Normalize the image data per channel for quicker and better training """
	
	def __call__(self, img_tensor):
		"""
		:param img_tensor: A single image tensor of shape (C x H x W)
		:return: Return normalized sample
		"""
		assert len(
			img_tensor.shape) == 3, '[Assertion Error]: Normalization is performed for each channel so input must be only one image data only'
		img_tensor = img_tensor.contiguous()
		mean_list = img_tensor.view(img_tensor.shape[0], -1).mean(dim=1)
		std_list = img_tensor.view(img_tensor.shape[0], -1).std(dim=1)
		
		normalizer = torch_transformer.Normalize(mean=mean_list, std=std_list)
		return normalizer(img_tensor)


