import h5py
import numpy as np
import shutil
import pandas as pd
import os
import cv2 as cv2
# import matplotlib.pyplot as plt

from misc_utils.tensor_sampling_utils import sample_tensors

# Helper function to sample the Bus class and create a wights file to our model
def sample_model_for_TR():

	weights_source_path = '/Users/royhirsch/Downloads/VGG_coco_SSD_512x512_iter_360000.h5'
	weights_destination_path = '/Users/royhirsch/Downloads/VGG_coco_SSD_512x512_iter_360000_2_classes.h5'

	# Make a copy of the weights file.
	shutil.copy(weights_source_path, weights_destination_path)

	# Load both the source weights file and the copy we made.
	# We will load the original weights file in read-only mode so that we can't mess up anything.
	weights_source_file = h5py.File(weights_source_path, 'r')
	weights_destination_file = h5py.File(weights_destination_path)

	classifier_names = ['conv4_3_norm_mbox_conf',
	                    'fc7_mbox_conf',
	                    'conv6_2_mbox_conf',
	                    'conv7_2_mbox_conf',
	                    'conv8_2_mbox_conf',
	                    'conv9_2_mbox_conf',
	                    'conv10_2_mbox_conf']

	# TODO: Set the number of classes in the source weights file. Note that this number must include
	#       the background class, so for MS COCO's 80 classes, this must be 80 + 1 = 81.
	n_classes_source = 81
	# TODO: Set the indices of the classes that you want to pick for the sub-sampled weight tensors.
	#       In case you would like to just randomly sample a certain number of classes, you can just set
	#       `classes_of_interest` to an integer instead of the list below. Either way, don't forget to
	#       include the background class. That is, if you set an integer, and you want `n` positive classes,
	#       then you must set `classes_of_interest = n + 1`.
	classes_of_interest = [0, 6]
	# classes_of_interest = 12 # Uncomment this in case you want to just randomly sub-sample the last axis instead of providing a list of indices.

	for name in classifier_names:
		# Get the trained weights for this layer from the source HDF5 weights file.
		kernel = weights_source_file[name][name]['kernel:0'].value
		bias = weights_source_file[name][name]['bias:0'].value

		# Get the shape of the kernel. We're interested in sub-sampling
		# the last dimension, 'o'.
		height, width, in_channels, out_channels = kernel.shape

		# Compute the indices of the elements we want to sub-sample.
		# Keep in mind that each classification predictor layer predicts multiple
		# bounding boxes for every spatial location, so we want to sub-sample
		# the relevant classes for each of these boxes.
		if isinstance(classes_of_interest, (list, tuple)):
			subsampling_indices = []
			for i in range(int(out_channels / n_classes_source)):
				indices = np.array(classes_of_interest) + i * n_classes_source
				subsampling_indices.append(indices)
			subsampling_indices = list(np.concatenate(subsampling_indices))
		elif isinstance(classes_of_interest, int):
			subsampling_indices = int(classes_of_interest * (out_channels / n_classes_source))
		else:
			raise ValueError("`classes_of_interest` must be either an integer or a list/tuple.")

		# Sub-sample the kernel and bias.
		# The `sample_tensors()` function used below provides extensive
		# documentation, so don't hesitate to read it if you want to know
		# what exactly is going on here.
		new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
		                                      sampling_instructions=[height, width, in_channels, subsampling_indices],
		                                      axes=[[3]],
		                                      # The one bias dimension corresponds to the last kernel dimension.
		                                      init=['gaussian', 'zeros'],
		                                      mean=0.0,
		                                      stddev=0.005)

		# Delete the old weights from the destination file.
		del weights_destination_file[name][name]['kernel:0']
		del weights_destination_file[name][name]['bias:0']
		# Create new datasets for the sub-sampled weights.
		weights_destination_file[name][name].create_dataset(name='kernel:0', data=new_kernel)
		weights_destination_file[name][name].create_dataset(name='bias:0', data=new_bias)

	# Make sure all data is written to our output file before this sub-routine exits.
	weights_destination_file.flush()

	conv4_3_norm_mbox_conf_kernel = weights_destination_file[classifier_names[0]][classifier_names[0]]['kernel:0']
	conv4_3_norm_mbox_conf_bias = weights_destination_file[classifier_names[0]][classifier_names[0]]['bias:0']

	print("Shape of the '{}' weights:".format(classifier_names[0]))
	print()
	print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
	print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)

# Helper function to generate a CSV file in the relevant format for the data loader
def drop_GT_notaions_to_scv(source_notaiton_file, dest_notaiton_file_path='', out_dim=512):

	# Get the scale relations for each axis
	org_h = 2736
	org_w = 3648

	scale_h = out_dim / float(org_h)
	scale_w = out_dim / float(org_w)

	processed_data = pd.DataFrame([], columns=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'])
	with open(source_notaiton_file) as fp:
		i = 0
		for line in fp:
			tmp = line.split(':')
			file_name = tmp[0]
			rois = list(tmp[1].replace('],[', ' ').replace('[', ' ').replace(']', ' ').split())
			rois = [list(map(int, item.split(','))) for item in rois]

			for roi in rois:

				# Original roi is in form : [xmin, ymin, width, height, class]
				# transform to [xmin, ymin, xmax, ymax, class] -> [xmin, ymin, xmin + width, ymin + height, class]
				roi_points = [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3], roi[4]]

				# Scale the coordinates to the resized image
				roi_scaled = [int(roi_points[0] * scale_w), int(roi_points[1] * scale_h),
				             int(roi_points[2] * scale_w), int(roi_points[3] * scale_h), roi_points[-1]]

				# Write to CSV in format:
				# ['image file name', 'xmin', 'ymin', 'xmax', 'ymax', 'class ID']
				temp_ls = []
				split = file_name.split('.')[0]
				reformat_file_name = split + '.jpg'
				temp_ls.append(reformat_file_name)
				# Create unified list in the desired length
				roi_row = temp_ls + roi_scaled
				processed_data.loc[i] = roi_row
				i += 1

	out_file_path = os.path.join(dest_notaiton_file_path, 'gt_rois.csv')
	processed_data.to_csv(out_file_path)
	print('Saved notation CSV file at {}'.format(out_file_path))

# Helper plot functino, rects is a np.array [N, [x1, x2, y1, y2]]
def drawRects(orgImg, rects, GTrects=None):

	# orgImg -> openCV file
	imOut = orgImg.copy()

	# Itereate over all the region proposals
	if len(np.shape(rects)) == 1:
		x1, y1, x2, y2 = rects
		cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
	else:
		for rect in rects:
			# Draw rectangle for region proposal till numShowRects
			x1, y1, x2, y2 = rect
			cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

	if GTrects:
		if len(np.shape(GTrects)) == 1:
			x1, y1, x2, y2 = GTrects
			cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
		else:
			for rect in GTrects:
				x1, y1, x2, y2 = rect
				cv2.rectangle(imOut, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)

	# show output
	plt.figure()
	plt.imshow(cv2.cvtColor(imOut, cv2.COLOR_BGR2RGB))
	plt.show()

# Gets bbox_pred_filter fro, detection net in format:
# [class, conf, xmin, ymin, xmax, ymax]
# Transforms into [xmin, ymin, w, h, class] and rescale to original image size
def resize_bbox_to_original(bbox_pred, scale_h, scale_w):
	# Filter the bbox format
	corners = bbox_pred[0][:, 2:]
	p_classes = bbox_pred[0][:, 0]

	reformat_bbox = []
	# Iterate over all the bbox prediction and reformat
	for i, corner in enumerate(corners):
		xmin = corner[0]
		ymin = corner[1]
		xmax = corner[2]
		ymax = corner[3]
		p_class = p_classes[i]
		new_cord = [xmin, ymin, xmax - xmin, ymax - ymin]

		new_cord = [int(new_cord[0] / scale_w),
		            int(new_cord[1] / scale_h),
		            int(new_cord[2] / scale_w),
		            int(new_cord[3] / scale_h),
					p_class]

		reformat_bbox.append(new_cord)
	# Return the coordinated as ints
	return np.array(reformat_bbox).astype(np.int32)

# Cuts the buses bbox from the original image and resize to desired dim
# Return a np array of shape [N, dim, dim, 4] that contains the resized buses crops.
def get_predicted_bbox_cropes(resize_bbox, org_im, out_dim):
	crops = np.zeros((len(resize_bbox), out_dim, out_dim, 3))

	# Crop and resize each bus bbox
	for i, bbox in enumerate(resize_bbox):
		bbox = bbox[1:]
		crop = org_im[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
		crop = cv2.resize(crop, (out_dim, out_dim))
		crops[i, :, :, :] = crop
	return crops

def quick_imshow_numpy(img):
	plt.figure()
	plt.imshow(img)
	plt.show()

'''
Test code fot image and BB resize:

im_root = '/Users/royhirsch/Documents/Study/Current/ComputerVision/project/busesTrain'
image_path = os.path.join(im_root, file_name)
im = cv2.imread(image_path)
r_im = cv2.resize(im, (512, 512))

#org image
drawRects(im, roi_points[:-1])
drawRects(r_im, roi_scaled[:-1])
'''

