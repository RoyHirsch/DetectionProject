import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
import sys
import cv2 as cv2
import pickle
import os
from utils import *

''''
    This is a script for preparation of data for train the NN model.
    The script generates region proposal and GT labels to the training data.
    
    The script uses openCV selective_search to extract region proposals from the data.
    Then, the region proposals are compared with the ground true label.
    A positive score is given to a region with IOU over 0.5.
    
    For each image a pickle file with the rects and the label is being saved.
    Moreover, the script saved the resized original images
'''

''' ###################################### PARAMETERS ###################################### '''
data_dir   = '/Users/royhirsch/Documents/GitHub/DetectionProject/busesTrain'
label_path = '/Users/royhirsch/Documents/GitHub/DetectionProject/annotationsTrain.txt'
scale      = 0.25
output_dir = '/Users/royhirsch/Documents/GitHub/DetectionProject/ProcessedData'

''' ###################################### FUNCTIONS ###################################### '''

def selective_search(im, method='reg'):

    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    # Switch to fast but low recall Selective Search method
    if method == 'fast':
        ss.switchToSelectiveSearchFast()

    # Switch to high recall but slow Selective Search method
    else:
        ss.switchToSelectiveSearchQuality()

    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    return rects

def intersection_over_union(rectA, rectB):
    x, y, h, w = rectA
    boxA = [x, y, x+w, y+h]
    x, y, h, w, c = rectB
    boxB = [x, y, x + w, y + h]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def btach_intersection_over_union(rectA, rectB):
    # rectA in a batch of rect
    # rectB my also contain many labels
    cls_label = []
    rg_label = []
    for itemA in rectA:
        grades = []
        for itemB in rectB:
            grades.append((intersection_over_union(itemA, itemB), itemB))
        if any([grade >= 0.5 for grade, _ in grades]):
            cls_label.append(1)
            rg_label.append(itemB)
        else:
            cls_label.append(0)
            rg_label.append([0, 0, 0, 0, 0])
    return np.array(cls_label), np.array(rg_label)

''' ###################################### MAIN SCRIPT ###################################### '''
# read labels file:
columns_list = ['image_name', 'rois']
labels_raw = pd.DataFrame([], columns=columns_list)
with open(label_path) as fp:
    for i, line in enumerate(fp):
        tmp = line.split(':')
        file_name = tmp[0]
        rois = list(tmp[1].replace('],[', ' ').replace('[', ' ').replace(']', ' ').split())
        rois = [list(map(int, item.split(','))) for item in rois]

        scale_rois = []
        for roi in rois:
            scale_roi = [int(num * scale) for num in roi[:-1]]
            scale_roi.append(roi[-1])
            scale_rois.append(scale_roi)
        labels_raw.loc[i] = [file_name, scale_rois]

# Read the images from the given dir:
for root, dirs, files in os.walk(data_dir):
    for i, name in enumerate(files):

        image_path = os.path.join(root, name)
        print('Start processing sample {}'.format(name))
        im = cv2.imread(image_path)

        # Resize image by scale
        newWidth = int(im.shape[0] * scale)
        newHeight = int(im.shape[1] * scale)
        im = cv2.resize(im, (newHeight, newWidth))

        rois = selective_search(im)

        # Create labels
        gt_labels = list(labels_raw.loc[labels_raw['image_name'] == name]['rois'])[0]
        cls_labels, reg_labels = btach_intersection_over_union(rois, gt_labels)
        # drawRects(im, rois, gt_labels) ### debug ###
        print('Number of positive labels: {}'.format(np.sum(cls_labels)))
        rois_n_labels = np.column_stack((rois, cls_labels, reg_labels))

        # Save as pickle file
        name_split = name.split('.')[0]
        saveStr = 'rois_for_' + str(name_split) +'.p'
        print('Ended to process sample {}'.format(name))
        pickle.dump(rois_n_labels, open(os.path.join(output_dir, saveStr), 'wb'))

        saveStr = 'resized_img_' + str(name_split) + '.p'
        pickle.dump(im, open(os.path.join(output_dir, saveStr), 'wb'))