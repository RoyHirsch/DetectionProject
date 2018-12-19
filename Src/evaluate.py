import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_loader import *
from RCNN import *

''' ############################ Parameters ############################'''
batch_size = 32
root_data_dir = '/Users/royhirsch/Documents/GitHub/ProcessedData'
checkpoint_path = '/Users/royhirsch/Documents/GitHub/DetectionProject/Checkpoints/checkpoint_epoch_20_val_loss_96.6224.pr'

def nms(dets, thresh):
    dets = dets.numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
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

    return torch.IntTensor(keep)

''' ############################    Main    ############################'''
net = RCNN()
net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

# Create test and validation data loaders
trainDataLoader = BusDataLoader(root_dir=os.path.join(root_data_dir, 'test'),
                                data_loader_type='test',
                                BGpad=16,
                                outShape=224,
                                balance_data_size=0,
                                augment_pos=0)

valDataLoader = BusDataLoader(root_dir=os.path.join(root_data_dir, 'validation'),
                              data_loader_type='validation',
                              BGpad=16,
                              outShape=224,
                              balance_data_size=0,
                              augment_pos=0)


test_loader = DataLoader(trainDataLoader, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(valDataLoader, batch_size=batch_size, shuffle=True, num_workers=4)


def accuracy(outputs, labels):
	'''
		Simple function to calculate the accuracy per minibatch
	'''
	_, preds = torch.max(outputs, 1)
	acc = (preds == labels).sum().float() * 100 / labels.shape[0]
	return acc.item()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device is ' + str(device))
net.to(device)
since = time.time()


# Initialize statistics per epoch
ls_acc = []
true_pos_samples = []
false_pos_samples = []

net.eval()
##### Evaluation loop for an epoch #####
for inputs, cls_labels, rg_labels, rects in test_loader:

	inputs = inputs.to(device)
	cls_labels = cls_labels.to(device)

	# forward
	outputs = net(inputs)
	for i in range(len(outputs)):
		if outputs[i] == 1:
			if cls_labels[i] == 1:
				# Document the TP
				# outputs[i][1] - prediction score, confidance at being bus
				true_pos_samples.append((rects[i], outputs[i][1], cls_labels[i]))
			else:
				# Document the FP
				false_pos_samples.append((rects[i], outputs[i][1], cls_labels[i]))
	# statistics
	ls_acc.append(accuracy(outputs, cls_labels))

	del inputs, cls_labels, rg_labels, rects, outputs
	torch.cuda.empty_cache()

print('Evaluation accuracy :: {:.4f} '.format(np.mean(ls_acc)))
print('True positive :: {:.4f}\nFalse positive :: {:.4f}'.format(len(true_pos_samples), len(false_pos_samples)))

rects = np.concatonate((true_pos_samples, false_pos_samples))
rects = rects[:5]
final = nms(rects, 0.5)