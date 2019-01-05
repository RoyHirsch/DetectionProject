import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


class RCNN(nn.Module):

	def __init__(self, nun_classes=2, num_regressions=4, train_phase='classification'):
		super(RCNN, self).__init__()

		self.nun_classes = nun_classes
		self.num_regressions = num_regressions
		self.train_phase = train_phase

		net_model = models.vgg16(pretrained=True)
		# Input size of last fc layer (4096)
		num_features = net_model.classifier[6].in_features

		# Freeze training for all layers
		for param in net_model.features.parameters():
			param.requires_grad = False

		for param in net_model.classifier.parameters():
			param.requires_grad = False

		# The net backbone
		self.vgg_features = nn.Sequential(*list(net_model.features._modules.values()))
		self.vgg_classifier = nn.Sequential(*list(net_model.classifier._modules.values())[:-1])

		# The net classification and regression heads
		self.classifier = nn.Linear(num_features, self.nun_classes)
		self.regression = nn.Linear(num_features, self.num_regressions)

	@staticmethod
	def flatten(x):
		input_shape = x.size()
		x = x.view(input_shape[0], int(input_shape[1] * input_shape[2] * input_shape[3]))
		return x

	def forward(self, x):
		x = self.vgg_features(x)
		x = self.flatten(x)
		x = self.vgg_classifier(x)

		if self.train_phase == 'classification':
			return self.classifier(x)

		elif self.train_phase == 'regressions':
			return self.regression(x)

		elif self.train_phase == 'both':
			return self.classifier(x), self.regression(x)

		else:
			raise ValueError('Invalide tarin phase: {}'.format(self.train_phase))
