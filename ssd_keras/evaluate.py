from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
from models.keras_ssd512 import ssd_512
import h5py
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from bounding_box_utils.bounding_box_utils import iou as iou
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights_path', type=str , default='', help='weights_path')
	parser.add_argument('--im_path', default='', type=str, help='im_path batch size')
	parser.add_argument('--test_label_path', default='', type=str, help='path to test labels csv')
	parser.add_argument('--val_label_path', default='', type=str, help='path to val labels csv')

	args = parser.parse_args()

	args_dict = vars(args)
	print('Model params are:')
	for k, v in args_dict.items():
		print(k + ' : ' + str(v))

	###############################################################################
	# 0: Pre-defined parameters
	###############################################################################

	# Data params
	batch_size = 1
	img_channels = 3
	# Do not change this value if you're using any of the pre-trained weights.
	mean_color = [123, 117, 104]
	swap_channels = [0, 1, 2]
	img_height = 512
	img_width = 512

	# Model params
	# Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
	n_classes = 1
	scales_pascal = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]
	scales_coco = [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
	# TODO: rethink about this param Roy
	scales = scales_coco
	aspect_ratios = [[1.0, 2.0, 0.5],
	                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
	                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
	                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
	                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
	                 [1.0, 2.0, 0.5],
	                 [1.0, 2.0, 0.5]]
	two_boxes_for_ar1 = True
	# The space between two adjacent anchor box center points for each predictor layer.
	steps = [8, 16, 32, 64, 128, 256, 512]
	# The offsets of the first anchor box center points from the top and left borders of the image
	# as a fraction of the step size for each predictor layer.
	offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
	# Whether or not to clip the anchor boxes to lie entirely within the image boundaries
	clip_boxes = False
	# The variances by which the encoded target coordinates are divided as in the original implementation
	variances = [0.1, 0.1, 0.2, 0.2]
	normalize_coords = True

	###############################################################################
	# 1: Functions
	###############################################################################

	def plot_predictions(orig_image, y_pred_thresh, gt_box):
		colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
		classes = ['background', 'bus']

		plt.figure(figsize=(20, 12))
		plt.imshow(orig_image)

		current_axis = plt.gca()

		for box in y_pred_thresh:
			# Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
			xmin = box[-4] * orig_image.shape[1] / img_width
			ymin = box[-3] * orig_image.shape[0] / img_height
			xmax = box[-2] * orig_image.shape[1] / img_width
			ymax = box[-1] * orig_image.shape[0] / img_height
			color = colors[int(box[0])]
			label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
			current_axis.add_patch(
				plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
			current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})

		for box in gt_box[0]:
			xmin = box[1]
			ymin = box[2]
			xmax = box[3]
			ymax = box[4]
			color = colors[10]
			label = 'GT'
			current_axis.add_patch(
				plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
			current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})

	# Reformat the bbox to the desired format and scale to original image size
	# Corners is [class, xmin, ymin, xmax, ymax] or [class, confidence, xmin, ymin, xmax, ymax]
	def reformat_box(corners, scale=True):
		if len(corners) == 6:
			cls = corners[0]
			corners = corners[2:]
		else:
			cls = corners[0]
			corners = corners[1:]

		xmin = corners[0]
		ymin = corners[1]
		xmax = corners[2]
		ymax = corners[3]
		new_cord = [xmin, ymin, xmax - xmin, ymax - ymin]

		if scale:
			dim = 512
			org_h = 2736
			org_w = 3648

			scale_h = dim / float(org_h)
			scale_w = dim / float(org_w)

			new_cord = [int(new_cord[0] / scale_w),
			            int(new_cord[1] / scale_h),
			            int(new_cord[2] / scale_w),
			            int(new_cord[3] / scale_h)]

		new_cord.append(cls)
		return new_cord

	###############################################################################
	# 2: Build the Keras model
	###############################################################################
	K.clear_session()  # Clear previous models from memory.
	print('Building the model')
	model = ssd_512(image_size=(img_height, img_width, img_channels),
	                n_classes=n_classes,
	                mode='inference',
	                l2_regularization=0.0005,
	                scales=scales,
	                aspect_ratios_per_layer=aspect_ratios,
	                two_boxes_for_ar1=two_boxes_for_ar1,
	                steps=steps,
	                offsets=offsets,
	                clip_boxes=clip_boxes,
	                variances=variances,
	                normalize_coords=normalize_coords,
	                subtract_mean=mean_color,
	                swap_channels=swap_channels,
	                confidence_thresh=0.5,
	                iou_threshold=0.45,
	                top_k=200,
	                nms_max_output_size=400)


	model.load_weights(args.weights_path, by_name=True)

	# 3: Compile the model so that Keras won't complain the next time you load it.
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
	model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

	###############################################################################
	# 3: Build the DataGenerator
	###############################################################################
	test_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
	test_dataset.parse_csv(images_dir=args.im_path,
	                        labels_filename=args.test_label_path,
	                        input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
	                        include_classes='all',
	                        random_sample=False,
	                        ret=False,
	                        verbose=True)

	val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
	val_dataset.parse_csv(images_dir=args.im_path,
	                      labels_filename=args.val_label_path,
	                      input_format=['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'],
	                      include_classes='all',
	                      random_sample=False,
	                      ret=False,
	                      verbose=True)

	# For the validation generator:
	convert_to_3_channels = ConvertTo3Channels()

	# Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
	predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
	                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
	                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
	                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
	                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
	                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
	                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

	ssd_input_encoder = SSDInputEncoder(img_height=img_height,
	                                    img_width=img_width,
	                                    n_classes=n_classes,
	                                    predictor_sizes=predictor_sizes,
	                                    scales=scales,
	                                    aspect_ratios_per_layer=aspect_ratios,
	                                    two_boxes_for_ar1=two_boxes_for_ar1,
	                                    steps=steps,
	                                    offsets=offsets,
	                                    clip_boxes=clip_boxes,
	                                    variances=variances,
	                                    matching_type='multi',
	                                    pos_iou_threshold=0.5,
	                                    neg_iou_limit=0.5,
	                                    normalize_coords=normalize_coords)

	# Create the generator handles that will be passed to Keras' `fit_generator()` function.
	test_generator = test_dataset.generate(batch_size=batch_size,
	                                         shuffle=False,
	                                         transformations=[],
	                                         label_encoder=ssd_input_encoder,
	                                         returns={'processed_images',
	                                                  'processed_labels',
	                                                  'filenames'},
	                                         keep_images_without_gt=False)

	val_generator = val_dataset.generate(batch_size=batch_size,
	                                     shuffle=False,
	                                     transformations=[],
	                                     label_encoder=ssd_input_encoder,
	                                     returns={'processed_images',
	                                              'processed_labels',
	                                              'filenames'},
	                                     keep_images_without_gt=False)

	test_dataset_size = test_dataset.get_dataset_size()
	val_dataset_size = val_dataset.get_dataset_size()

	print("Number of images in the test dataset:\t{:>6}".format(test_dataset_size))
	print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

	###############################################################################
	# 3: Predict !
	###############################################################################

	###############################################
	# Choose the data set to predict:
	gen = test_generator
	dataset_size = test_dataset_size
	###############################################

	im_name_ls = []
	gt_label_ls = []
	input_images_tensor = np.zeros([val_dataset_size, img_height, img_width, img_channels])
	for i in range(dataset_size):
		img, gt_label, im_name = next(gen)
		input_images_tensor[i, :, :, :] = img
		im_name_ls.append(im_name)
		gt_label_ls.append(gt_label)

	y_pred = model.predict(input_images_tensor)

	confidence_threshold = 0.5
	y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

	# Plot predictions
	for i in range(dataset_size):
		plot_predictions(input_images_tensor[i, :, :, :].astype(np.uint8), y_pred_thresh[i], gt_label_ls[i])

	# Print IOU scores:
	for i in range(dataset_size):
		print('Image name : {}'.format(im_name_ls[i][0]))
		print('Num of gt buses : {}'.format(len(gt_label_ls[i][0])))
		if len(gt_label_ls[i][0]) > 1:
			a = gt_label_ls[i][0][:, 1:]
		else:
			a = gt_label_ls[i][0][0][1:]

		if len(y_pred_thresh[i]) == 0:
			print('No IOU')
			continue

		if len(y_pred_thresh[i]) > 1:
			b = y_pred_thresh[i][:, 2:]
		else:
			b = y_pred_thresh[i][0][2:]

		iou_score = iou(np.array(a), np.array(b), coords='corners', mode='outer_product', border_pixels='half')
		if len(iou_score) == 0:
			print('No IOU')
		else:
			for row in iou_score:
				print('IOU is {} '.format(max(row)))