import numpy as np
import ast
import os

from keras import backend as K
from keras.optimizers import Adam
import cv2 as cv2
import numpy as np
import sys

# Import relevat outer files
sys.path.append(os.path.join(os.path.abspath(__file__ + '../../../'), 'ssd_keras'))
sys.path.append(os.path.join(os.path.abspath(__file__ + '../../../'), 'util_functions.py'))

from params import *
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from util_functions import resize_bbox_to_original

def run(myAnnFileName, buses):
    K.clear_session()
    # Instantiate the model
    model = ssd_512(image_size=(params['img_height'], params['img_width'], params['img_channels']),
                    n_classes=params['n_classes'],
                    mode=params['mode'],
                    l2_regularization=params['reg'],
                    scales=params['scales'],
                    aspect_ratios_per_layer=params['aspect_ratios'],
                    two_boxes_for_ar1=params['two_boxes_for_ar1'],
                    steps=params['steps'],
                    offsets=params['offsets'],
                    clip_boxes=params['clip_boxes'],
                    variances=params['variances'],
                    normalize_coords=params['normalize_coords'],
                    subtract_mean=params['mean_color'],
                    swap_channels=params['swap_channels'],
                    confidence_thresh=params['confidence_thresh'],
                    iou_threshold=params['iou_threshold'],
                    top_k=params['top_k'],
                    nms_max_output_size=params['nms_max_output_size'])

    # Load pre-trained weights
    model.load_weights(params['weights_path'], by_name=True)

    # Compile the model
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    str_output = ''
    for root, dirs, files in os.walk(buses):
        # Sort files in ascending way
        files.sort()

        # Iterate over all the images in the given root
        for name in files:
            image_full_path = os.path.join(root, name)
            org_im = cv2.imread(image_full_path)
            org_im = cv2.cvtColor(org_im, cv2.COLOR_BGR2RGB)

            org_h, org_w, c = np.shape(org_im)
            scale_h = params['img_height'] / float(org_h)
            scale_w = params['img_height'] / float(org_w)

            # Resize to the net desired input size
            im = cv2.resize(org_im, (params['img_height'], params['img_height']))
            bbox_pred = model.predict(np.expand_dims(im, axis=0))

            bbox_pred_filter = [bbox_pred[k][bbox_pred[k, :, 1] > params['confidence_thresh']] for k in
                                range(bbox_pred.shape[0])]
            if len(bbox_pred_filter) > 0:
                # Resize to the original size
                pred_bboxs = resize_bbox_to_original(bbox_pred_filter, scale_h, scale_w)

                # Write results into file
                temp_str = name + ':'
                for pred_bbox in pred_bboxs:
                    tmp_bbox_str = '['
                    for item in pred_bbox[:-1]:
                        tmp_bbox_str += str(item) + ', '
                    tmp_bbox_str += str(pred_bbox[-1]) + ']'
                    temp_str += tmp_bbox_str + ','
                temp_str = temp_str[:-1]
            str_output += temp_str + '\n'

        # Save results to txt file
        with open(myAnnFileName, 'w+') as f:
            f.write(str_output)
        print('Finished prediction, results as be fount at ....')
