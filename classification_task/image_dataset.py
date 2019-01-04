# -*-coding: utf-8 -*-

import os

import torch
from torch.utils.data import Dataset, DataLoader

import math
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.color import rgba2rgb


import pandas as pd

import util_cv as util


##
class ImageDataset(Dataset):
	""" Image dataset for custom reading of images """
	
	def __init__(self, datatype_dir, img_list_fname, transformer=None, has_label=True):
		"""
		Args:
			:param root_dir: Path of CSV file with labels
			:param transformer: to apply transformer on the image
		"""
		
		self.root_dir = util.BASE_DATAPATH
		self.datatype_dir = datatype_dir
		
		self.files_name = pd.read_csv(os.path.join(self.root_dir, self.datatype_dir, img_list_fname), delimiter=',',
		                              skipinitialspace=True, comment='#')
		self.transformer = transformer
		
		self.has_label = has_label
	
	def __len__(self):
		return len(self.files_name)
	
	def __getitem__(self, idx):
		
		img_name = self.files_name.iloc[idx, 0]
		img_fullpath = os.path.join(self.root_dir, self.datatype_dir, img_name)
		
		img_ndarray = io.imread(img_fullpath)
		
		if len(img_ndarray.shape) == 3:
			if img_ndarray.shape[2] == 4:
				img_ndarray = rgba2rgb(img_ndarray)
		else:  # If b&w image repeat the dimension
			img_ndarray = np.expand_dims(img_ndarray, axis=2)
			img_ndarray = np.concatenate((img_ndarray, img_ndarray, img_ndarray), axis=2)
		
		if self.transformer:
			img_ndarray = self.transformer(img_ndarray)
		
		if self.has_label:
			label = self.files_name.iloc[idx, 1]
			return img_ndarray, label
		else:
			return img_ndarray, None