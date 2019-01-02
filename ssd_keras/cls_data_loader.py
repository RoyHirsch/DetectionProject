import numpy as np
from keras.models import Sequential
import numpy as np
import pandas as pd
import cv2 as cv2
import os
import keras
from skimage import transform
import matplotlib.pyplot as plt

def random_rotation(im):
	random_degree = np.random.uniform(-30, 30)
	return transform.rotate(im, random_degree)

def horz_flip(im):
	return cv2.flip(im, 0)

def vert_flip(im):
	return cv2.flip(im, 0)

def rand_augment(im):
	trans = [random_rotation, horz_flip, vert_flip]

	num_trans = np.random.randint(0, len(trans))
	inds = np.random.choice(len(trans), num_trans, replace=False)

	for i in inds:
		im = trans[i](im)
	return im

class DataGenerator(keras.utils.Sequence):
	def __init__(self, root_image_dir, label_path, batch_size=32, dim=64,
	             n_channels=3, n_classes=6, shuffle=True):

		print('Initialized datalaoder obj')
		self.root_image_dir = root_image_dir
		self.label_path = label_path
		self.dim = dim
		self.batch_size = batch_size
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle

		# self.org_h = 2736
		# self.org_w = 3648
		# self.scale_h = self.dim / float(self.org_h)
		# self.scale_w = self.dim / float(self.org_w)

		self.gt_images = self._get_raw_images()
		self.gt_labels = self._get_gt_labels()
		self.gt_buses = self._extract_buses()

		self.on_epoch_end()

	def _extract_buses(self):
		buses_df = pd.DataFrame([], columns=['image_name', 'bus_crop', 'class_num'])

		# Iterate over all the images
		k = 0
		print('Start extracting bus crops')
		for i in range(len(self.gt_images)):
			im_name, im = self.gt_images.loc[i]
			_, bbxox_n_class = self.gt_labels.loc[self.gt_labels['image_name'] == im_name + '.JPG'].values[0]

			# Iterate over all the bbox in the image
			for bbxox in bbxox_n_class:
				xmin = bbxox[0]
				ymin = bbxox[1]
				w = bbxox[2]
				h = bbxox[3]
				cls = bbxox[4]

				crop = im[ymin:ymin + h, xmin:xmin + w, :]
				crop = cv2.resize(crop, (self.dim, self.dim))
				buses_df.loc[k] = [im_name, crop, cls]
				k += 1
		print('Extracted {} buses crops'.format(k))
		return buses_df

	def _get_raw_images(self):
		data_frame = pd.DataFrame([], columns=['image_name', 'image'])
		i = 0
		for root, dirs, files in os.walk(self.root_image_dir):
			for name in files:
				split_name = name.replace('.', '_').split('_')

				image_path = os.path.join(root, name)
				im = cv2.imread(image_path)

				# im = cv2.resize(im, (self.dim, self.dim))

				# Convert from BGR to RGB (reverse the depth dim order)
				img = im[..., ::-1]

				data_frame.loc[i] = [split_name[0], img]
				i += 1

		return data_frame

	def _get_gt_labels(self):
		labels_raw = pd.DataFrame([], columns=['image_name', 'gt_np_array'])
		with open(self.label_path) as fp:
			for i, line in enumerate(fp):
				tmp = line.split(':')
				file_name = tmp[0]
				rois = list(tmp[1].replace('],[', ' ').replace('[', ' ').replace(']', ' ').split())
				rois = [list(map(int, item.split(','))) for item in rois]

				# scale_rois = []
				# for roi in rois:
				#     # Original roi is in form : [xmin, ymin, width, height, class]
				#     # transform to [xmin, ymin, xmax, ymax, class] -> [xmin, ymin, xmin + width, ymin + height, class]
				#     roi_points = [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3], roi[4]]
				#
				#     # Scale the coordinates to the resized image
				#     roi_scaled = [int(roi_points[0] * self.scale_w), int(roi_points[1] * self.scale_h),
				#                   int(roi_points[2] * self.scale_w), int(roi_points[3] * self.scale_h), roi_points[-1]]
				#
				#     scale_rois.append(roi_scaled)

				# Good np format for the augmentation package
				np_rois = np.array(rois)
				labels_raw.loc[i] = [file_name, np_rois]
		return labels_raw

	@staticmethod
	def conv_rect_to_opos_points(rect):
		if len(rect) == 4:
			return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
		else:
			return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3], rect[4]]

	def __len__(self):
		'Denotes the number of batches per epoch'
		return len(self.gt_images)

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = np.random.choice(self.indexes, self.batch_size)

		# Generate data
		X, y = self.__data_generation(indexes)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.gt_images))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, indexes):
		'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
		# Initialization

		X = np.zeros([self.batch_size, self.dim,  self.dim, self.n_channels]).astype(np.uint8)
		y = np.zeros([self.batch_size, self.n_classes]).astype(np.uint8)

		# Generate data batch
		for num, ind in enumerate(indexes):
			# Store sample
			im_name, crop, cls = self.gt_buses.loc[ind]
			X[num, :, :, :] = rand_augment(crop)
			y[num, :] = keras.utils.to_categorical(cls-1, num_classes=self.n_classes)

		return X, y
