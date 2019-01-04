# -*- coding: utf-8 -*-


import os
import sys

import numpy
import skimage
from skimage import io, transform

import torch


# Project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data dir
BASE_DATAPATH = os.path.join(PROJECT_DIR, 'data')
TRAIN_DATAPATH = os.path.join(BASE_DATAPATH, 'training')
TEST_DATAPATH = os.path.join(BASE_DATAPATH, 'testing')

# Image Dimension
HEIGHT_DEFAULT_VAL = 256
WIDTH_DEFAULT_VAL = 256
BATCH_SIZE = 2

# Store trained model
TRAINED_MODELPATH = os.path.join(PROJECT_DIR, 'models')
TRAINED_MODELNAME = 'conv_net.model'



# For logging:
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


def reset_logger(filename='train_output.log'):
	logger.handlers = []
	filepath = os.path.join(PROJECT_DIR, filename)
	logger.addHandler(logging.FileHandler(filepath, 'w'))


def add_logger(filename):
	filepath = os.path.join(PROJECT_DIR, filename)
	logger.addHandler(logging.FileHandler(filepath, 'w'))


def setup_logger(filename='output.log'):
	filepath = os.path.join(PROJECT_DIR, filename)
	logger.addHandler(logging.FileHandler(filepath, 'a'))











# For device agnostic training/testing: https://pytorch.org/blog/pytorch-0_4_0-migration-guide/
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def load_training_data():
	train_data_path = TRAIN_DATAPATH
	print('Train data-path = ', train_data_path)
	
	dirs = [dir for dir in os.listdir(train_data_path) if not dir.startswith('.')]
	print(dirs)
	
	x_data = []
	y_target = []
	for dir in dirs:
		current_dir_label = int(dir)
		files = [file for file in os.listdir(os.path.join(train_data_path, dir)) if '.png' in file.lower()]
		for file in files:
			image = io.imread(os.path.join(train_data_path, dir, file))
			
		
	


if __name__ == '__main__':
	load_training_data()