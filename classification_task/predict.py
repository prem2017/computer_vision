# -*- coding: utf-8 -*-

import os


import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as torch_transformer

import torchvision
from skimage import io
import skimage.transform as sk_transformer

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import util_cv as util
from train import *


import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


def load_trained_model():
	model_path = os.path.join(util.TRAINED_MODELPATH, util.TRAINED_MODELNAME)
	
	pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
	
	untrained_model = ConvNet()
	
	# In case model changes since the last time trained model was saved
	# untrained_model_dict = untrained_model.state_dict()
	# pretrained_dict_insection_with_untrained_model_dict = pretrained_dict = {
	# 	k: v for k, v in pretrained_dict.items() if k in untrained_model_dict}
	
	untrained_model.load_state_dict(pretrained_dict)
	# Put the model in eval mode
	untrained_model.eval()
	
	return untrained_model



def compute_prob(y_pred):
	with torch.no_grad():
		softmax = nn.Softmax(dim=1)
		# y_pred = torch.from_numpy(y_pred).contiguous()
		output_prob = softmax(y_pred).data
		return output_prob.numpy()


def generate_report(model, data_loader, image_names=None):
	
	
	y_true_all = None
	y_pred_all = None
	with torch.no_grad():
		for i, xy in enumerate(data_loader):
			print('[Predicting for] i = ', i)
			img_tensor, label = xy
			img_tensor, label  = img_tensor.float(), label.long()
			img_tensor = img_tensor.to(util.device)
			
			y_pred = model(img_tensor)
			y_pred = compute_prob(y_pred)
			if label is not None: # if label is not none
				label = label.numpy().reshape(-1,)
				if y_true_all is None:
					y_true_all = label
				else:
					y_true_all = np.hstack((y_true_all, label))
			
			if y_pred_all is None:
				y_pred_all = y_pred
			else:
				y_pred_all = np.vstack((y_pred_all, y_pred))
				

	_, ypred_labels_all = torch.max(torch.from_numpy(y_pred_all), 1)
	
	ypred_labels_all = ypred_labels_all.numpy()
	
	
	
	
	if y_true_all is not None:
		clf_report = classification_report(y_true_all, ypred_labels_all)
		conf_mat = confusion_matrix(y_true_all, ypred_labels_all)
		msg = '\n\n[Classification Report] = \n{}\n\n[Confusion Matrix] = \n{}'.format(clf_report, conf_mat)
		logger.info(msg); print(msg)
	msg = '\n\n[Output] = \n{}'.format(ypred_labels_all)
	logger.info(msg); print(msg)
	
	# output prediction in csv
	if image_names is not None:
		header = ['ImageName', 'PredictedProb_0', 'PredictedProb_1']
		image_names = image_names.reshape(-1, 1)
		y_pred_all = np.round(y_pred_all, 3)
		df = np.hstack((image_names, y_pred_all))
		
		if y_true_all is not None:
			header.append('true_lable')
			df = np.hstack((df, y_true_all.reshape(-1, 1)))
		
		
		df = pd.DataFrame(df, columns=header)
		df.to_csv(path_or_buf=os.path.join(util.PROJECT_DIR, 'test_output_prediction.csv'), sep=',', index=None, header=header)

	






if __name__ == '__main__':
	print('Run <predict.py>')
	
	util.reset_logger('predcition_output.log')

	# predict_dir, img_list_fname, has_labels = False)
	model = load_trained_model()
	model = model.to(util.device)
	
	test_dir = 'testing'
	img_list_fname = 'test_iminfo.csv' # has_labels
	
	image_names = pd.read_csv(os.path.join(util.BASE_DATAPATH, test_dir, img_list_fname)).values
	image_names = image_names[:, 0]
	
	has_label = True
	
	transformer = torch_transformer.Compose([RescaleImage((util.HEIGHT_DEFAULT_VAL, util.WIDTH_DEFAULT_VAL)),
	                                         ToTensor(),
	                                         NormalizeImageData()
	])
	
	img_dataset = ImageDataset(datatype_dir=test_dir,
	                           img_list_fname=img_list_fname,
	                           transformer=transformer,
	                           has_label=has_label)
	
	test_dataloader = DataLoader(dataset=img_dataset,
	                             batch_size=1,
	                             shuffle=False)
	
	
	generate_report(model, test_dataloader, image_names=image_names)
	