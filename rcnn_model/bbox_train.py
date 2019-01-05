import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import sys
import time
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_loader import *
from RCNN import *

def train_model(model, criterion, optimizer, scheduler, dataloaders, save_dir, num_epochs=25):

	def accuracy(outputs, rects, labels):
		'''
			Simple function to calculate the accuracy per minibatch
		'''

		preds = torch.stack((rects[:, 2]*outputs[:, 0]+rects[:, 0], rects[:, 3]*outputs[:, 1]+rects[:, 1],
				rects[:, 2]*torch.from_numpy(np.exp(outputs[:, 2].detach().numpy())), rects[:, 3]*torch.from_numpy(np.exp(outputs[:, 3].detach().numpy()))),1)
		acc = (abs(preds-labels[:, :-1].float())<10).sum().float()*100 / (4*labels.shape[0])
		return acc.item()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Device is ' + str(device))
	model.to(device)
	since = time.time()

	# Saves a tupal of (train_acc, val_acc per epoch)
	ls_epoch_acc = []
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Initialize statistics per epoch
		ls_acc_train = []
		ls_loss_train = []
		ls_loss_val = []
		ls_acc_val = []

		scheduler.step()
		model.train()  # Set model to training mode

		##### Train loop for an epoch #####
		for inputs, cls_labels, rg_labels, rects in tqdm(dataloaders['train'], desc='Train epoch %d' % epoch):
			start_epoch = time.time()
			inputs = inputs.to(device)
			rg_labels = rg_labels.to(device)
			rects = rects.to(device)
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward
			outputs = model(inputs)
			G = rg_labels[:, :-1]
			P = rects
			t = torch.stack(((G.float()[:, 0]-P[:, 0])/P[:, 2], (G.float()[:, 1]-P[:, 1])/P[:, 3],
				np.log(G.float()[:, 2]/P[:, 2]), np.log(G.float()[:, 3]/P[:, 3])), 1)
			loss = criterion(outputs, t)
			# print('iter loss =', loss.item())

			# backward + optimize only if in training phase
			loss.backward()
			optimizer.step()

			# statistics
			ls_loss_train.append(loss.item())
			ls_acc_train.append(accuracy(outputs, rects, rg_labels))

			del inputs, cls_labels, rg_labels, rects, outputs
			torch.cuda.empty_cache()

		stop_epoch = time.time() - start_epoch
		print('Train :: Loss: {:.4f} Acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(
			np.mean(ls_loss_train), np.mean(ls_acc_train), stop_epoch // 60, stop_epoch % 60))

		##### Evaluate over validation data #####
		if epoch % 5 == 0:
			model.eval()
			for inputs, cls_labels, rg_labels, rects in tqdm(dataloaders['val'], desc='Validation epoch %d' % epoch):
				inputs = inputs.to(device)
				rg_labels = rg_labels.to(device)
				rects = rects.to(device)
				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				outputs = model(inputs)
				G = rg_labels[:, :-1]
				P = rects
				t = torch.stack(((G.float()[:, 0] - P[:, 0]) / P[:, 2], (G.float()[:, 1] - P[:, 1]) / P[:, 3],
					np.log(G.float()[:, 2] / P[:, 2]), np.log(G.float()[:, 3] / P[:, 3])), 1)
				loss = criterion(outputs, t)

				# statistics
				ls_loss_val.append(loss.item())
				ls_acc_train.append(accuracy(outputs, rects, rg_labels))

				del inputs, cls_labels, rg_labels, rects
				torch.cuda.empty_cache()

			current_epoch_acc = np.mean(ls_acc_val)
			print('Validation :: Loss: {:.4f} Acc: {:.4f}'.format(np.mean(ls_loss_val), current_epoch_acc))
			ls_epoch_acc.append((np.mean(ls_acc_train), current_epoch_acc))

			# Simple method for saving parameters (if current val_acc better than the previous two)
			if epoch >= 10 and current_epoch_acc > ls_epoch_acc[-2][1] and current_epoch_acc > ls_epoch_acc[-3][1]:
				filename = 'checkpoint_epoch_{}_val_loss_{}.pr'.format(epoch, str(round(current_epoch_acc, 4)))
				checkpoint_path = os.path.join(save_dir, filename)
				torch.save(model.state_dict(), filename)
				print('Saved model checkpoint with val_acc: {}'.format(str(round(current_epoch_acc, 4))))

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	return model


''' ############################ Parameters ############################'''
lr = 0.001

batch_size = 32
# root_data_dir should contain 3 sub-folders: 'train' , 'validation' and  'test
root_data_dir = '/Users/Mor\'s Yoga/Documents/GitHub/DetectionProject/ProcessedData'

''' ############################    Main    ############################'''
# net = create_model()
net = RCNN(num_regressions=4, train_phase='regressions')
# net.load_state_dict(torch.load(PATH_TO_STATE_DICT_PR_FILE))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1000)

# Decay LR by a factor of 0.1 * lr every n epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.0001)

# Create train and validation data loaders
trainDataLoader = BusDataLoader(root_dir=os.path.join(root_data_dir, 'train'),
                                data_loader_type='train',
                                BGpad=16,
                                outShape=224,
                                balance_data_size=1,
                                augment_pos=0)

valDataLoader = BusDataLoader(root_dir=os.path.join(root_data_dir, 'validation'),
                              data_loader_type='validation',
                              BGpad=16,
                              outShape=224,
                              balance_data_size=0,
                              augment_pos=0)

# Sample only pos samples
train_ind = trainDataLoader.get_ind_of_positive_samples()
train_pos_sampler = SubsetRandomSampler(train_ind)
val_ind = valDataLoader.get_ind_of_positive_samples()
val_pos_sampler = SubsetRandomSampler(val_ind)

# Pay attention to num_workers , if there are problems -> num_workers=0
train_loader = DataLoader(trainDataLoader, batch_size=batch_size, shuffle=False, sampler=train_pos_sampler, num_workers=4)
val_loader = DataLoader(valDataLoader, batch_size=batch_size, shuffle=False, sampler=val_pos_sampler, num_workers=4)

print('\n################## Begin training ! ################## ')
print('Number of batches in train epoch is {}'.format(len(train_loader)))

dataDict = {'train': train_loader, 'val': val_loader}
model_ft = train_model(net, criterion, optimizer, exp_lr_scheduler, dataDict, save_dir='', num_epochs=50)

# To save the model state dict
torch.save(model_ft.state_dict(), '')



