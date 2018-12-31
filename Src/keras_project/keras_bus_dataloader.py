import numpy as np
from keras.models import Sequential
import numpy as np
import pandas as pd
import cv2 as cv2
import os
import keras
import matplotlib.pyplot as plt

class DataGenerator(keras.utils.Sequence):

    def __init__(self, root_image_dir, label_path, batch_size=32, dim=512,
                 n_channels=3,n_classes=2, shuffle=True):

        print('Initialized datalaoder obj')
        self.root_image_dir = root_image_dir
        self.label_path = label_path
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.org_h = 2736
        self.org_w = 3648

        self.scale_h = self.dim / float(self.org_h)
        self.scale_w = self.dim / float(self.org_w)

        self.gt_images = self._get_raw_images()
        self.gt_labels = self._get_gt_labels()

        self.on_epoch_end()


    def _get_raw_images(self):
        data_frame = pd.DataFrame([], columns=['image_name', 'image'])
        i = 0
        for root, dirs, files in os.walk(self.root_image_dir):
            for name in files:
                split_name = name.replace('.', '_').split('_')

                image_path = os.path.join(root, name)
                im = cv2.imread(image_path)

                im = cv2.resize(im, (self.dim, self.dim))

                # Convert from BGR to RGB (reverse the depth dim order)
                img = im[..., ::-1]

                data_frame.loc[i] = [split_name[0], img]
                i += 1

        return data_frame

    def _get_gt_labels(self):
        labels_raw = pd.DataFrame([], columns=['image_name', 'gt_np_array', 'gt_list_cls'])
        with open(self.label_path) as fp:
            for i, line in enumerate(fp):
                tmp = line.split(':')
                file_name = tmp[0]
                rois = list(tmp[1].replace('],[', ' ').replace('[', ' ').replace(']', ' ').split())
                rois = [list(map(int, item.split(','))) for item in rois]

                scale_rois = []
                for roi in rois:
                    # Original roi is in form : [xmin, ymin, width, height, class]
                    # transform to [xmin, ymin, xmax, ymax, class] -> [xmin, ymin, xmin + width, ymin + height, class]
                    roi_points = [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3], roi[4]]

                    # Scale the coordinates to the resized image
                    roi_scaled = [int(roi_points[0] * self.scale_w), int(roi_points[1] * self.scale_h),
                                  int(roi_points[2] * self.scale_w), int(roi_points[3] * self.scale_h), roi_points[-1]]

                    scale_rois.append(roi_scaled)

                # Good np format for the augmentation package
                np_rois = np.array(scale_rois)[:, :-1]
                labels_raw.loc[i] = [file_name, np_rois, scale_rois]
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
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.zeros([self.batch_size, self.dim, self.dim, self.n_channels]).astype(np.uint8)
        y = []

        # Generate data batch
        for num, ind in enumerate(indexes):
            # Store sample
            im_name, X[num, :, :, :] = self.gt_images.loc[ind]
            im_name, bbox_only, bbxox_n_class = self.gt_labels.loc[self.gt_labels['image_name'] == im_name + '.JPG'].values[0]
            y.append(bbox_only)

        return X, y


root_data_dir = '/Users/royhirsch/Documents/Study/Current/ComputerVision/project/busesTrain'
label_path = '/Users/royhirsch/Documents/Study/Current/ComputerVision/project/annotationsTrain.txt'

dg = DataGenerator(root_data_dir, label_path)
dg.__getitem__(1)
