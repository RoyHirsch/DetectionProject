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

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# %matplotlib inline


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights_path', type=str , default='', help='weights_path')
	parser.add_argument('--im_path', default='', type=str, help='im_path batch size')

	args = parser.parse_args()

	# Set the image size.
	img_height = 512
	img_width = 512
	# 1: Build the Keras model

	K.clear_session()  # Clear previous models from memory.

	model = ssd_512(image_size=(img_height, img_width, 3),
	                n_classes=1,
	                mode='inference',
	                l2_regularization=0.0005,
	                scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
	                # The scales for MS COCO are [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
	                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
	                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
	                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
	                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
	                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
	                                         [1.0, 2.0, 0.5],
	                                         [1.0, 2.0, 0.5]],
	                two_boxes_for_ar1=True,
	                steps=[8, 16, 32, 64, 128, 256, 512],
	                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
	                clip_boxes=False,
	                variances=[0.1, 0.1, 0.2, 0.2],
	                normalize_coords=True,
	                subtract_mean=[123, 117, 104],
	                swap_channels=[0,1,2],
	                confidence_thresh=0.5,
	                iou_threshold=0.45,
	                top_k=600,
	                nms_max_output_size=500)

	# 2: Load the trained weights into the model.
	# TODO: Set the path of the trained weights.
	# weights_path = 'path/to/trained/weights/VGG_VOC0712_SSD_300x300_iter_120000.h5'

	weights_source_file = h5py.File(args.weights_path, 'r')

	model.load_weights(args.weights_path, by_name=True)

	# 3: Compile the model so that Keras won't complain the next time you load it.

	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

	model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

	im = cv2.imread(args.im_path)[:,:,::-1]
	orig_images = np.expand_dims(im, axis=0)

	orig_image = cv2.resize(im, (img_height, img_width))
	input_images = np.expand_dims(orig_image, axis=0)

	print('input image is {} and shape {} X {}'.format(type(input_images),np.shape(input_images)[0],np.shape(input_images)[1]))

	y_pred = model.predict(input_images)
	confidence_threshold = 0.5

	y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

	np.set_printoptions(precision=2, suppress=True, linewidth=90)
	print("Predicted boxes:\n")
	print('   class   conf xmin   ymin   xmax   ymax')
	print(y_pred_thresh[0])

	# Display the image and draw the predicted boxes onto it.
	#
	# # Set the colors for the bounding boxes
	# colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
	# classes = ['background','bus']
	#
	# plt.figure(figsize=(20, 12))
	# plt.imshow(orig_images[0])
	#
	# current_axis = plt.gca()
	#
	# for box in y_pred_thresh[0]:
	# 	# Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
	# 	xmin = box[-4] * orig_images[0].shape[1] / img_width
	# 	ymin = box[-3] * orig_images[0].shape[0] / img_height
	# 	xmax = box[-2] * orig_images[0].shape[1] / img_width
	# 	ymax = box[-1] * orig_images[0].shape[0] / img_height
	# 	color = colors[int(box[0])]
	# 	label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
	# 	current_axis.add_patch(
	# 		plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
	# 	current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
	# print('t')
	#
	#
	#
	#
	#
