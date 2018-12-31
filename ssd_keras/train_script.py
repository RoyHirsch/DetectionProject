from keras import backend as K
from keras.optimizers import Adam
import argparse
from models.keras_ssd512 import ssd_512
import h5py
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger


if __name__ == '__main__':
	# Parse parameters to run the script
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights_path', type=str, default='', help='path to modified weights file')
	parser.add_argument('--im_path', default='', type=str, help='path to root images dir')
	parser.add_argument('--train_label_path', default='', type=str, help='path to train labels csv')
	parser.add_argument('--val_label_path', default='', type=str, help='path to val labels csv')
	parser.add_argument('--num_epochs', default=10, type=int, help='num of epochs to train')
	parser.add_argument('--steps_per_epoch', default=500, type=int, help='staps per epoch')
	parser.add_argument('--batch_size', default=32, type=int, help='batch size')
	parser.add_argument('--lr', default=0.00001, type=float, help='lr default is 0.001')

	args = parser.parse_args()

	args_dict = vars(args)
	print('Model params are:')
	for k, v in args_dict.items():
		print(k + ' : ' + str(v))

	###############################################################################
	# 0: Pre-defined parameters
	###############################################################################

	# Data params
	batch_size = args.batch_size
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
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
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
	# 1: Build the Keras model
	###############################################################################
	K.clear_session()  # Clear previous models from memory.
	print('Building the model')
	model = ssd_512(image_size=(img_height, img_width, img_channels),
	                n_classes=n_classes,
	                mode='training',
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

	#Load the trained weights into the model.
	weights_source_file = h5py.File(args.weights_path, 'r')
	model.load_weights(args.weights_path, by_name=True)

	#Compile the model
	adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
	model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

	###############################################################################
	# 2: Build the DataGenerator
	###############################################################################
	print('Building the DataGenerators')
	train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
	train_dataset.parse_csv(images_dir=args.im_path,
                  labels_filename=args.train_label_path,
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

	# Set the image transformations for pre-processing and data augmentation options.
	ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
	                                            img_width=img_width,
	                                            background=mean_color)
	# For the validation generator:
	convert_to_3_channels = ConvertTo3Channels()
	# resize = Resize(height=img_height, width=img_width)

	#Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
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
	train_generator = train_dataset.generate(batch_size=batch_size,
	                                         shuffle=True,
	                                         transformations=[ssd_data_augmentation],
	                                         label_encoder=ssd_input_encoder,
	                                         returns={'processed_images',
	                                                  'encoded_labels'},
	                                         keep_images_without_gt=False)

	val_generator = val_dataset.generate(batch_size=batch_size,
	                                     shuffle=True,
	                                     transformations=[ssd_data_augmentation],
	                                     label_encoder=ssd_input_encoder,
	                                     returns={'processed_images',
	                                              'encoded_labels'},
	                                     keep_images_without_gt=False)

	train_dataset_size = train_dataset.get_dataset_size()
	val_dataset_size = val_dataset.get_dataset_size()

	print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
	print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))
	###############################################################################
	# 3: Define model callbacks.
	###############################################################################
	model_checkpoint = ModelCheckpoint(
		filepath='ssd_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
		monitor='val_loss',
		verbose=1,
		save_best_only=True,
		save_weights_only=True, # Changed to save only weights
		mode='auto',
		period=1)
	# model_checkpoint.best =

	csv_logger = CSVLogger(filename='training_log.csv',
	                       separator=',',
	                       append=True)

	# learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
	#                                                 verbose=1)

	terminate_on_nan = TerminateOnNaN()

	callbacks = [model_checkpoint]
	# csv_logger,
	#  learning_rate_scheduler,
	#  terminate_on_nan]

	###############################################################################
	# 4: Train
	###############################################################################
	initial_epoch = 0
	steps_per_epoch = args.steps_per_epoch
	history = model.fit_generator(generator=train_generator,
	                              steps_per_epoch=steps_per_epoch,
	                              epochs=args.num_epochs,
	                              callbacks=callbacks,
	                              # workers=3,
	                              # use_multiprocessing=True,
	                              validation_data=val_generator,
	                              validation_steps=200, #TODO its' just a number...
	                              initial_epoch=initial_epoch)

'''
Code to validate the data generator:
import matplotlib.pyplot as plt
classes = ['background', 'bus']

batch_size = 1

data_generator = train_dataset.generate(batch_size=batch_size,
                                  shuffle=False,
                                  transformations=[ssd_data_augmentation],
                                  label_encoder=None,
                                  returns={'processed_images',
                                           'processed_labels',
                                           'filenames',
                                           'original_images',
                                           'original_labels'},
                                  keep_images_without_gt=False)
                                  
processed_images, processed_annotations, filenames, original_images, original_annotations = next(data_generator)

colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist() # Set the colors for the bounding boxes
i=0
fig, cell = plt.subplots(1, 2, figsize=(20,16))
cell[0].imshow(original_images[i])
cell[1].imshow(processed_images[i])

for box in original_annotations[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    color = colors[int(box[0])]
    label = '{}'.format(classes[int(box[0])])
    cell[0].add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    cell[0].text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
    
for box in processed_annotations[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    color = colors[int(box[0])]
    label = '{}'.format(classes[int(box[0])])
    cell[1].add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    cell[1].text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

'''
