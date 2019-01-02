import numpy as np
import ast
import os

from keras import backend as K
from keras.optimizers import Adam
import cv2 as cv2
import numpy as np
import sys
sys.path.append(os.path.join(os.path.abspath(__file__ + '../../../'), 'ssd_keras'))
sys.path.append(os.path.join(os.path.abspath(__file__ + '../../../'), 'util_functions.py'))
from params import *
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from util_functions import resize_bbox_to_original, get_predicted_bbox_cropes

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
model.load_weights(defs['weights_path'], by_name=True)

# Compile the model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# Iterate over all the imsges in the given root
for root, dirs, files in os.walk('/Users/royhirsch/Documents/Study/Current/ComputerVision/project/busesTrain'):
    for name in files:
        image_full_path = os.path.join(root, name)
        org_im = cv2.imread(image_full_path)

        org_h, org_w, c = np.shape(org_im)
        scale_h = params['img_height'] / float(org_h)
        scale_w = params['img_height'] / float(org_w)

        im = cv2.resize(org_im, (params['img_height'], params['img_height']))
        bbox_pred = model.predict(np.expand_dims(im, axis=0))

        bbox_pred_filter = [bbox_pred[k][bbox_pred[k, :, 1] > params['confidence_thresh']] for k in range(bbox_pred.shape[0])]
        if len(bbox_pred_filter) > 0:
            resize_bbox = resize_bbox_to_original(bbox_pred_filter, scale_h, scale_w)
            bbox_crops = get_predicted_bbox_cropes(resize_bbox, org_im, out_dim=128)

    # plt.imshow(bbox_crops[0, :, :, :].astype(np.uint8))


# def run(myAnnFileName, buses):
#
#     annFileNameGT = os.path.join(os.getcwd(),'annotationsTrain.txt')
#     writtenAnnsLines = {}
#     annFileEstimations = open(myAnnFileName, 'w+')
#     annFileGT = open(annFileNameGT, 'r')
#     writtenAnnsLines['Ground_Truth'] = (annFileGT.readlines())
#
#     for k, line_ in enumerate(writtenAnnsLines['Ground_Truth']):
#
#         line = line_.replace(' ','')
#         imName = line.split(':')[0]
#         anns_ = line[line.index(':') + 1:].replace('\n', '')
#         anns = ast.literal_eval(anns_)
#         if (not isinstance(anns, tuple)):
#             anns = [anns]
#         corruptAnn = [np.round(np.array(x) + np.random.randint(low = 0, high = 100, size = 5)) for x in anns]
#         corruptAnn = [x[:4].tolist() + [anns[i][4]] for i,x in enumerate(corruptAnn)]
#         strToWrite = imName + ':'
#         if(3 <= k <= 5):
#             strToWrite += '\n'
#         else:
#             for i, ann in enumerate(corruptAnn):
#                 posStr = [str(x) for x in ann]
#                 posStr = ','.join(posStr)
#                 strToWrite += '[' + posStr + ']'
#                 if (i == int(len(anns)) - 1):
#                     strToWrite += '\n'
#                 else:
#                     strToWrite += ','
#         annFileEstimations.write(strToWrite)
