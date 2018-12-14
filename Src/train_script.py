import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import sys
import time
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from data_loader import *

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Device is ' + str(device))
	since = time.time()

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		running_loss_train = 0.0
		running_corrects_train = 0
		running_loss_val = 0.0
		running_corrects_val = 0

		scheduler.step()
		model.train()  # Set model to training mode

		##### Train loop for an epoch #####
		for inputs, labels, rects in dataloaders['train']:
			inputs = inputs.to(device)
			labels = labels.to(device)
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)

			loss = criterion(outputs , labels)
			# TODO: temporaly print of step loss (Roy)
			print('Step loss: {}'.format(loss.item()))

			# backward + optimize only if in training phase
			loss.backward()
			optimizer.step()

			# statistics
			running_loss_train += loss.item() * inputs.size(0)
			running_corrects_train += torch.sum(preds == labels.data)

			del inputs, labels, outputs, preds
			torch.cuda.empty_cache()

		epoch_train_loss = running_loss_train / dataloaders['train'].__len__()
		epoch_train_acc = running_corrects_train.double() / dataloaders['train'].__len__()

		print('Train :: Loss: {:.4f} Acc: {:.4f}'.format(epoch_train_loss, epoch_train_acc))

		##### Evaluate over validation data #####
		model.eval()
		for inputs, labels, rects in dataloaders['val']:
			inputs = inputs.to(device)
			labels = labels.to(device)

			# Only forward - predict test outputs
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)

			loss = criterion(outputs , labels)

			# statistics
			running_loss_val += loss.item() * inputs.size(0)
			running_corrects_val += torch.sum(preds == labels.data)

			del inputs, labels, outputs, preds
			torch.cuda.empty_cache()

		epoch_val_loss = running_loss_val / dataloaders['val'].__len__()
		epoch_val_acc = running_corrects_val.double() / dataloaders['test'].__len__()
		print('Validation :: Loss: {:.4f} Acc: {:.4f}'.format(epoch_val_loss, epoch_val_acc))

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	return model

def create_model(backbone='VGG16', num_class=2):
	'''

	:param backbone: currently only VGG16 will be generalized
	:param num_class:
	:return:
	'''
	if backbone == 'VGG16':
		net_model = models.vgg16(pretrained=True)

	else:
		raise ValueError('Invalid backbone net model')

	# Freeze training for all layers
	for param in net_model.features.parameters():
		param.requires_grad = False

	for param in net_model.classifier.parameters():
		param.requires_grad = False

	# Input size of last fc layer (4096)
	num_features = net_model.classifier[6].in_features
	# Remove last layer
	features = list(net_model.classifier.children())[:-1]
	# Add our layer with 4 outputs
	features.extend([nn.Linear(num_features, num_class)])
	# Replace the model classifier
	net_model.classifier = nn.Sequential(*features)

	print('Created {} model:'.format(backbone))
	# print(net_model)

	return net_model

def create_random_subsamples(size_dataset, test_par=0.2):
	'''
	creates SubsetRandomSampler object per train and test dataloaders

	:param size_dataset: the total number of samples in the dataset
	:param test_par: between [0,1] parentage of data to split for test
	:return:
	'''
	ind = list(range(size_dataset))
	test_size = int(test_par * size_dataset)

	test_ind = np.random.choice(ind, size=test_size, replace=False)
	train_ind = list(set(ind) - set(test_ind))

	train_sampler = SubsetRandomSampler(train_ind)
	test_sampler = SubsetRandomSampler(test_ind)

	return train_sampler, test_sampler

''' ############################ Parameters ############################'''
lr = 0.0001
batch_size = 32
# root_data_dir should contain 3 sub-folders: 'train' , 'validation' and  'test
root_data_dir = '/Users/royhirsch/Documents/GitHub/DetectionProject/ProcessedData'

''' ############################    Main    ############################'''
net = create_model()
# weight_tensor = torch.tensor([1, 100]).float()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Decay LR by a factor of 0.1 every n epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Create train and validation data loaders
trainDataLoader = BusDataLoader(root_dir=os.path.join(root_data_dir, 'train'),
                                data_loader_type='train',
                                BGpad=16,
                                outShape=224,
                                balance_data_size=1,
                                augment_pos=2)

valDataLoader = BusDataLoader(root_dir=os.path.join(root_data_dir, 'validation'),
                              data_loader_type='validation',
                              BGpad=16,
                              outShape=224,
                              balance_data_size=0,
                              augment_pos=0)

# Pay attention to num_workers , if there are problems -> num_workers=0
train_loader = DataLoader(trainDataLoader, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(valDataLoader, batch_size=4, shuffle=True, num_workers=4)

print('\n################## Begin training ! ################## ')
print('Number of batches in train epoch is {}\nMay the force be with you!'.format(len(train_loader)))

dataDict = {'train': train_loader, 'val': val_loader}
model_ft = train_model(net, criterion, optimizer, exp_lr_scheduler, dataDict, num_epochs=25)

# To save the model state dict
torch.save(model_ft.state_dict(), path='')



