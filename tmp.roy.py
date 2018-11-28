import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import sys
import time
import copy

from data_loader import *

num_class = 2
vgg16_model = models.vgg16(pretrained=True)

# Freeze training for all layers
for param in vgg16_model.features.parameters():
	param.require_grad = False

vgg16_feature_map = nn.Sequential(*list(vgg16_model.features._modules.values())[:-1])

# num_features = vgg16_model.classifier[6].in_features         # input size of last fc layer (4096)
# features = list(vgg16_model.classifier.children())[:-1]      # Remove last layer
# features.extend([nn.Linear(num_features, len(num_class))])   # Add our layer with 4 outputs
# vgg16_model.classifier = nn.Sequential(*features)            # Replace the model classifier
# print(vgg16_model)
# criterion = nn.CrossEntropyLoss()
#
# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(vgg16_model.parameters(), lr=0.001, momentum=0.9)
#
# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Dataloaders:
# TrainDataLoader = BusDataLoader(root_dir='/Users/royhirsch/Documents/Study/Current/ComputerVision/project/busesTrain',
#                                 label_path='/Users/royhirsch/Documents/Study/Current/ComputerVision/project/annotationsTrain.txt',
#                                 transform=data_transform)
#
# TestDataLoader = BusDataLoader(root_dir='/Users/royhirsch/Documents/Study/Current/ComputerVision/project/busesTrain',
#                                 label_path='/Users/royhirsch/Documents/Study/Current/ComputerVision/project/annotationsTrain.txt',
#                                 transform=data_transform)
#
# model_ft = train_model(vgg16_model, criterion, optimizer_ft, exp_lr_scheduler, [TrainDataLoader, TestDataLoader],
# 					   num_epochs=25)
# To save the model state dict
# torch.save(model_ft.state_dict(), path='')

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			ind = 0 if phase == 'train' else 1
			epoch_loss = running_loss / dataloaders[ind].__len__()
			epoch_acc = running_corrects.double() / dataloaders[ind].__len__()

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model


