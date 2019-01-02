# Params dict for the whole model
params =      {'batch_size': 1,
               'img_channels': 3,
               # Do not change this value if you're using any of the pre-trained weights.
               'mean_color': [123, 117, 104],
			   'swap_channels': [0, 1, 2],
			   'img_height': 512,
			   'img_width': 512,

			   # Model params
			   # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
			   'n_classes': 6,
               'mode': 'inference',
               'reg': 0.0005,
			   'scales_pascal': [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
			   'scales': [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06],
			   'aspect_ratios': [[1.0, 2.0, 0.5],[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                  [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5],
                                  [1.0, 2.0, 0.5]],
			   'two_boxes_for_ar1': True,
			   'steps': [8, 16, 32, 64, 128, 256, 512],
			   'offsets': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
			   'clip_boxes': False,
			   'variances': [0.1, 0.1, 0.2, 0.2],
	           'normalize_coords': True,
	           'confidence_thresh': 0.5,
			   'iou_threshold': 0.45,
               'top_k': 200,
               'nms_max_output_size':400,
               # TODO: need to change to the local weights_path
               'weights_path': '/Users/royhirsch/Downloads/ssd_epoch_7_classes-03_loss-2.0023_val_loss-1.9532.h5'}