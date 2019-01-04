# -*-coding: utf-8 -*-

import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as torch_transformer
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR


import math
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.color import rgba2rgb
import skimage.transform as sk_transformer

import pandas as pd

import util_cv as util

from models import *
from image_dataset import *
from transformer import *
from loss_functions import *


import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()







# Train the network
def train_network(dataloader, model, loss_function, optimizer, start_lr, end_lr, num_epochs=90):
	model = model.train()
	logger_msg = '\nDataLoader = %s' \
	             '\nModel = %s' \
	             '\nLossFucntion = %s' \
	             '\nOptimizer = %s' \
	             '\nStartLR = %s, EndLR = %s' \
	             '\nNumEpochs = %s' %(dataloader, model, loss_function, optimizer, start_lr, end_lr, num_epochs)
	
	logger.info(logger_msg), print(logger_msg)

	# [https://arxiv.org/abs/1803.09820]
	# This is used to find optimal learning-rate which can be used in one-cycle training policy
	# lr_scheduler = MultiStepLR(optimizer=optimizer,
	#                            milestones=list(np.arange(2, 24, 2)),
	#                            gamma=10, last_epoch=-1)
	
	def get_lr():
		lr = []
		for param_group in optimizer.param_groups:
			lr.append(np.round(param_group['lr'], 11))
		return lr
	
	def set_lr(lr):
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		
	
	
	
	total_len = 0
	current_epoch_batchwise_loss = []
	avg_epoch_loss_container = [] # Stores loss for each epoch averged over
	
	all_epoch_batchwise_loss = []
	
	extra_epochs = 10
	total_epochs = num_epochs + extra_epochs
	
	# One cycle setting of Learning Rate
	num_steps_upndown = 10
	further_lowering_factor = 5
	further_lowering_factor_step = 2
	def one_cycle_lr_setter(current_epoch):
		if current_epoch <= num_epochs:
			lr_inc_rate = (end_lr - start_lr) / (2 * num_steps_upndown)
			lr_inc_epoch_step_len = int(num_epochs / (2 * num_steps_upndown))
			
			steps_completed = int(current_epoch / lr_inc_epoch_step_len)
			if steps_completed < num_steps_upndown:
				current_lr = start_lr - (steps_completed * lr_inc_rate)
			else:
				current_lr = end_lr + ((steps_completed - num_steps_upndown) * lr_inc_rate)
			set_lr(current_lr)
		else:
			current_lr = end_lr / (further_lowering_factor**(((current_epoch - num_epochs) / further_lowering_factor_step)+1))
			set_lr(current_lr)
			
		
	for epoch in range(total_epochs):
		msg = '\n\n\n[Epoch] = %s' %(epoch+1)
		print(msg)
		for i, data in enumerate(dataloader):
			x, y = data
			x, y = x.float(), y.long()
			x, y = x.to(device=util.device), y.to(device=util.device)
			
			optimizer.zero_grad()
			
			y_pred = model(x)
			loss = loss_function(y_pred, y)
			
			loss.backward()
			optimizer.step()
			
			current_epoch_batchwise_loss.append(loss.item())
			all_epoch_batchwise_loss.append(loss.item())
			
			batch_run_msg = '\nEpoch: [%s/%s], Step: [%s/%s], InitLR: %s, CurrentLR: %s, Loss: %s' \
			                %(epoch + 1, total_epochs, i+1,  len(dataloader), start_lr, get_lr(), loss.item())
			print(batch_run_msg)
		# store average loss
		avg_epoch_loss = np.round(sum(current_epoch_batchwise_loss) / (i+1.0), 6)
		avg_epoch_loss_container.append(avg_epoch_loss)
		current_epoch_batchwise_loss = []
		if avg_epoch_loss < 1e-6 or get_lr()[0] < 1e-9 or get_lr()[0] > 10:
			msg = '\n\nAvg. Loss = {} or Current LR = {} thus stopping training'.format(avg_epoch_loss, get_lr())
			logger.info(msg); print(msg)
			break
		
		# TODO: Only for estimating good learning rate
		# lr_scheduler.step(epoch+1)
		one_cycle_lr_setter(epoch+1)
	
	# Print the loss
	msg = '\n\n[Epoch Loss] = {}'.format(avg_epoch_loss_container)
	logger.info(msg); print(msg)
	
	# Plot
	plot_loss(avg_epoch_loss_container, plot_file_name='training_avg_epoch_loss.png', title='Training Epoch Loss')
	plot_loss(all_epoch_batchwise_loss, plot_file_name='training_batchwise.png', title='Training Batchwise Loss', xlabel='#Batchwise')
	
	# Save the model
	save_model(model)
	
	
	
# Plot training loss
def plot_loss(training_loss, plot_file_name='training_loss.png', title='Training Loss', xlabel='Epochs'):
	
	fig = plt.figure()
	
	plt.plot(range(len(training_loss)), training_loss, '-*',  markersize=6, lw=3, alpha=0.8)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('Cross Entropy Loss')
	
	
	full_path = os.path.join(util.PROJECT_DIR, plot_file_name)
	fig.tight_layout()  # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
	fig.savefig(full_path)
	plt.close(fig)  # clo
	
	
	


def save_model(model):
	model_path = os.path.join(util.TRAINED_MODELPATH, 'conv_net.model')
	
	if next(model.parameters()).is_cuda:
		model = model.cpu().float()
	
	model_dict = model.state_dict()
	torch.save(model_dict, model_path)
	
	model = model.to(util.device)
	return model

# def train_network(dataloader, model, loss_function, optimizer, start_lr, end_lr, num_epochs=100):
# Pre-requisite setup for training process
def train_convnet(datatype_dir, img_list_fname):
	"""
	Setup the environment for training such as model, loss function, optimizer e.t.c.
	:return: None
	"""
	
	train_params={}
	train_params['start_lr'] = start_lr = 7e-4 # 1e-7
	train_params['end_lr'] = 11e-4 # 10
	train_params['num_epochs'] = 80
	
	dropout = 0.5
	weight_decay = 1e-4
	
	model = ConvNet(dropout=dropout)
	train_params['model'] = model = model.to(util.device)
	
	loss_function = ClassificationLoss()
	train_params['loss_function'] = loss_function.to(util.device)
	
	train_params['optimizer'] = optim.Adam(params=model.parameters(),
	                                       lr=start_lr,
	                                       weight_decay=weight_decay)
	# optim.SGD(params=model.parameters(), lr=start_lr, weight_decay=weight_decay, momentum=0.9)
	
	# Rescale the image if it is big and then randomly crop as they still keep the type of data to detect
	# Can also use <Rotation>, <Pixel-Shifting> for data augmentation
	# TODO: Rescaling is OK but make sure that cropping is center
	transformer = torch_transformer.Compose([RescaleImage((int(1.2 * util.HEIGHT_DEFAULT_VAL), int(1.2 * util.WIDTH_DEFAULT_VAL))),
	                                         RandomCropImage((util.HEIGHT_DEFAULT_VAL, util.WIDTH_DEFAULT_VAL)),
	                                         ToTensor(),
	                                         NormalizeImageData()
	])
	dataset = ImageDataset(datatype_dir=datatype_dir,
	                       img_list_fname=img_list_fname,
	                       transformer=transformer)
	train_params['dataloader'] = DataLoader(dataset=dataset,
	                                        batch_size=util.BATCH_SIZE,
	                                        shuffle=True)
	
	# Train the network
	train_network(**train_params)
	
	





if __name__ == '__main__':
	print('hello')
	torch.manual_seed(999)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(999)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	
	util.reset_logger()
	
	train_convnet(datatype_dir='training', img_list_fname='training_iminfo.csv')
	msg = '\n\n********************** Training Complete **********************\n\n'
	logger.info(msg); print(msg)