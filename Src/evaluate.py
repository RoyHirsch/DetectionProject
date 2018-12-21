import torch
import sys
import os
import math
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np
import time
from prepare_data_script import *
from data_loader import *
from RCNN import *

''' ############################ Parameters ############################'''
batch_size = 32
root_data_dir = '/Users/royhirsch/Documents/GitHub/ProcessedData'
checkpoint_path = '/Users/royhirsch/Downloads/checkpoint_epoch_5_val_loss_96.3821.pr'
scale = 0.25
nms_thres = 0.5

''' ############################ Functions ############################'''


def nms(dets, thresh):
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 0] + dets[:, 2]
	y2 = dets[:, 1] + dets[:, 3]
	scores = dets[:, 4]

	areas = (x2 - x1 + 1) * (y2 - y1 + 1)

	# Descending sort, start from max score
	order = scores.argsort()[::-1]

	keep = []
	# Start from the highest score, if there is an intersected rect - exclude it
	while order.size > 0:
		i = order.item(0)
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])

		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		ovr = inter / (areas[i] + areas[order[1:]] - inter)

		inds = np.where(ovr <= thresh)[0]
		order = order[inds + 1]

	return dets[keep, :]


def accuracy(outputs, labels):
	'''
		Simple function to calculate the accuracy per minibatch
	'''
	_, preds = torch.max(outputs, 1)
	acc = (preds == labels).sum().float() * 100 / labels.shape[0]
	return acc.item()


def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def statistics(outputs, labels):
	'''
		1 - true class - bus
		0 - background
	'''
	_, preds = torch.max(outputs, 1)

	preds = preds.detach().numpy()
	labels = labels.detach().numpy()

	TP = np.sum(np.logical_and(preds == 1, labels == 1))
	TN = np.sum(np.logical_and(preds == 0, labels == 0))
	FP = np.sum(np.logical_and(preds == 1, labels == 0))
	FN = np.sum(np.logical_and(preds == 0, labels == 1))

	per = TP / float(TP + FP)
	rec = TP / float(TP + FN)
	F1 = 2 * (per * rec) / (per + rec)

	return per, rec, F1


''' ############################    Main    ############################'''
net = RCNN()
net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

# Create test and validation data loaders
testDataLoader = BusDataLoader(root_dir=os.path.join(root_data_dir, 'test'),
                               data_loader_type='test',
                               BGpad=16,
                               outShape=224,
                               balance_data_size=2,
                               augment_pos=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device is ' + str(device))
net.to(device)

# Initialize statistics data structures
ls_acc = []
results = []
F1_ls = []
per_ls = []
rec_ls = []
all_res_dict = {'F1': [], 'per': [], 'rec': []}
net.eval()

##### Evaluation loop for an epoch #####
print('Start evaluation')

image_name_ls = list(set(testDataLoader.rect_data['image_name'].tolist()))
for name in image_name_ls:
	since = time.time()
	print('\nEvaluate image :: {}'.format(name))
	res = np.where(testDataLoader.rect_data['image_name'] == name)
	row_inds = res[0].tolist()
	sampler = SubsetRandomSampler(row_inds)

	single_image_dataloader = DataLoader(testDataLoader, batch_size=batch_size, sampler=sampler)

	with tqdm(total=len(single_image_dataloader), file=sys.stdout) as pbar:
		for inputs, cls_labels, rg_labels, rects in single_image_dataloader:
			pbar.set_description('Validation')
			pbar.update(1)

			inputs = inputs.to(device)
			cls_labels = cls_labels.to(device)

			# forward
			outputs = net(inputs)
			_, preds = torch.max(outputs, 1)

			# statistics
			per, rec, F1 = statistics(outputs, cls_labels)
			F1_ls.append(F1)
			per_ls.append(per)
			rec_ls.append(rec)
			ls_acc.append(accuracy(outputs, cls_labels))

			for i in range(len(outputs)):
				if preds[i] == 1:
					# Get the pos predicted results
					results.append((rects[i].detach().numpy(), outputs[i][1].detach().numpy(), rg_labels[i].numpy()))

			del inputs, cls_labels, rg_labels, rects, outputs
			torch.cuda.empty_cache()

	time_elapsed = time.time() - since
	print('Evaluation accuracy :: {:.4f} '.format(np.mean(ls_acc)))
	print('Precision :: {:.4f}\nRecall :: {:.4f}\nF1 :: {:.4f}'.format(np.mean(per_ls), np.mean(rec_ls), np.mean(F1_ls)))
	all_res_dict['F1'] = np.mean(F1_ls)
	all_res_dict['per'] = np.mean(per_ls)
	all_res_dict['rec'] = np.mean(rec_ls)

	# The arrange the data for NMS
	# Compress the network score with sigmoid
	filtered_rects = []
	for rect in results:
		filtered_rects.append(rect[0].tolist() + [sigmoid(rect[1].tolist())])

	# NMS
	final_rects = nms(np.array(filtered_rects), nms_thres)
	print('Evaluation of {} complete in {:.0f}m {:.0f}s'.format(name, time_elapsed // 60, time_elapsed % 60))

# Show the results
# final_rects_view = final_rects[:, :-1].astype(np.int32)
# img_ind = np.where(testDataLoader.resized_images['image_name'] == name)
# img = testDataLoader.resized_images.iloc[img_ind]['image']
# drawRects(img.values[0], final_rects_view, GTrects=None, numShowRects=len(final_rects_view))

print('Overall evaluation:')
print('Precision :: {:.4f}\nRecall :: {:.4f}\nF1 :: {:.4f}'.format(np.mean(all_res_dict['per']),
                                                                   np.mean(all_res_dict['rec']),
                                                                   np.mean(all_res_dict['F1'])))
