import torch
from torch.utils.data import Dataset, DataLoader
import re
import pandas as pd
import numpy as np
from torchvision import transforms, datasets
import os
from PIL import Image
import sys
import cv2 as cv2
import pickle
# from utils import *

class BusDataLoader(Dataset):
    def __init__(self, root_dir, data_loader_type='', BGpad=16, outShape=225, balance_data_size=2, augment_pos=4):
        """
        DataLoader class for handling and pre-processing of the data
        This class handles picked files containing pre-found region of interest rects that
        where found by selective search algorithm.

        Args:
            root_dir (string)                       : Path to the root folder of the pickeled files
            data_loader_type (string)               : (optional) description of the dataloader,
                                                      may be [train, validation, test]
            BGpad (int, must be equal)              : (optional) How much background padding to add
            outShape (int)                          : The desired input shape to the BB NN

            balance_data_size (int)                 : (optional) defines the pos:neg ratio for
                                                      balancing the data.
                                                      balance_data_size is the num of pos per one neg sample
            augment_pos (int)                       : (optional) define the number of augmentations per positive sample
                                                      if greater than 0, there will be augmentations

        Methods:
        __getitem__ - generates the relevant data of single example:
                      warped_image - a fixed size warped image to enter the NN backbone model
                      label        - GT label (binary)
                      rect         - [x, y, w, h] array represents the original bounding box
                                     (in the original image coordinates)
                                     (x, y) - top left corner of the bounding box
        __len__     - number of data samples in the class

        """
        self.root_dir = root_dir
        self.data_loader_type = data_loader_type
        self.BGpad = BGpad
        self.balance_data_size = balance_data_size
        self.outShape = outShape
        self.augment_pos = augment_pos

        self.rect_data = self._process_data()

        if self.augment_pos:
            self._augment_pos()

        if self.balance_data_size:
            self._balance_data()

        value_count = self.rect_data['label'].value_counts().to_dict()
        print('Data summery for {} data loader:\nPositive {}\nNegative {}'.format(self.data_loader_type,
                                                                                 value_count[1],
                                                                                 value_count[0]))

        self.resized_images = self._process_images()

        self.transform = transforms.Compose([
                         transforms.ToPILImage(),
                         transforms.Resize(size=[self.outShape, self.outShape]),
                         transforms.ToTensor()])

    def _augment_pos(self):
        '''
        If self.augment_pos != 0 search for all the positive samples and augment them.
        append all the augmentations to the self.rect_data
        '''
        columns_list_rois = ['image_name', 'rect', 'label']
        permutation_number = self.augment_pos
        pos_rows = self.rect_data.loc[self.rect_data['label'] == 1]

        print('Permutating {} positive rows for {} permutations each.'.format(len(pos_rows), permutation_number))
        perm_data_frame = pd.DataFrame([], columns=columns_list_rois)

        # Constants:
        # Dimensions of the image after resize
        org_h = 684
        org_w = 912

        min_change_parentage = 5. / 100
        max_change_parentage = 10. / 100

        min_change_x, max_change_x = int(min_change_parentage * org_h), int(max_change_parentage * org_h)
        min_change_y, max_change_y = int(min_change_parentage * org_w), int(max_change_parentage * org_w)

        # run for every pos row and do permutation_number permutations
        for row in pos_rows.iterrows():
            x, y, h, w = row[1]['rect']
            im_name = row[1]['image_name']

            perm_list = []
            for num in range(permutation_number):
                xt = x + np.random.choice([-1, 1], 1) * np.random.randint(min_change_x, max_change_x,1)
                yt = y + np.random.choice([-1, 1], 1) * np.random.randint(min_change_y, max_change_y,1)

                xt = x if xt < 0 else xt
                yt = y if yt < 0 else yt

                scale = np.random.choice([0, 1], 1)
                if scale:
                    wt = w + np.random.choice([-1, 1], 1) * np.random.randint(min_change_x, max_change_x, 1)
                    ht = h + np.random.choice([-1, 1], 1) * np.random.randint(min_change_y, max_change_y, 1)

                    wt = w if wt < 0 else wt
                    ht = h if ht < 0 else ht

                else:
                    wt = w
                    ht = h

                perm_list.append([im_name, [int(xt), int(yt), int(wt), int(ht)], 1])

            temp = pd.DataFrame(perm_list, columns=columns_list_rois)
            perm_data_frame = perm_data_frame.append(temp, ignore_index=True)

        # Append all the permutations
        self.rect_data = self.rect_data.append(perm_data_frame, ignore_index=True)

        # Shuffle the dataFrame
        self.rect_data = self.rect_data.sample(frac=1).reset_index(drop=True)

    def _balance_data(self):
        '''
        If self.balance_data_size != 0 delete some of the neg samples in order to maintain a specific
        retio of pos:neg samples
        '''
        value_count = self.rect_data['label'].value_counts().to_dict()
        num_pos = value_count[1]
        num_neg = int(num_pos * self.balance_data_size)
        num_of_drop_rows = value_count[0] - num_neg
        print('Dropped {} rows of negative samples out ot total {} rows'.format(num_of_drop_rows, value_count[0]))

        # sort temp - true label first
        self.rect_data = self.rect_data.sort_values(by=['label'], ascending=False)
        self.rect_data.drop(self.rect_data.tail(num_of_drop_rows).index, inplace=True)

        # Shuffle the dataFrame
        self.rect_data = self.rect_data.sample(frac=1).reset_index(drop=True)

    def _process_data(self, columns_list_rois=['image_name', 'rect', 'label']):
        data_frame = pd.DataFrame([], columns=columns_list_rois)

        for root, dirs, files in os.walk(os.path.join(self.root_dir, 'rects')):
            for name in files:
                if re.findall(r'rois_for_.*', name):
                    split_name = name.replace('.', '_').split('_')
                    rects = pickle.load(open(os.path.join(self.root_dir, 'rects', name), "rb"))

                    temp = pd.DataFrame([[split_name[2], rect[:-1], rect[-1]] for rect in rects], columns=columns_list_rois)
                    data_frame = data_frame.append(temp, ignore_index=True)

        return data_frame

    def _process_images(self, columns_list=['image_name', 'image']):
        data_frame = pd.DataFrame([], columns=columns_list)
        i = 0

        for root, dirs, files in os.walk((os.path.join(self.root_dir, 'images'))):
            for name in files:
                if re.findall(r'resized_img_.*', name):
                    split_name = name.replace('.', '_').split('_')
                    img = pickle.load(open(os.path.join(self.root_dir, 'images', name), "rb"))
                    # Convert from BGR to RGB (reverse the depth dim order)
                    img = img[...,::-1]

                    data_frame.loc[i] = [split_name[2], img]
                    i += 1
        return data_frame

    def __len__(self):
        return len(self.rect_data)

    # Getter, normalizes the data and the labels
    def __getitem__(self, idx):
        # Get the data of a rect from the main table
        name, rect, label = self.rect_data.loc[idx]

        # Extract the relevant image from the images table
        img = self.resized_images.loc[self.resized_images['image_name'] == name]['image'].values[0]

        # Pad the bounding box with background pixels before warping
        if self.BGpad:
            BGpad = self.BGpad
            h, w, c = np.shape(img)
            minX = rect[1] - BGpad//2 if (rect[1] - BGpad//2) >=0 else 0
            maxX = rect[1] + rect[3] + BGpad // 2 if (rect[1] + rect[3] + BGpad // 2) <= h else h

            minY = rect[0] - BGpad // 2 if (rect[0] - BGpad // 2) >= 0 else 0
            maxY = rect[0] + rect[2] + BGpad // 2 if (rect[0] + rect[2] + BGpad // 2) <= w else w
            crop_img = img[minX:maxX, minY:maxY, :]

        else:
            # crop the image by rect
            crop_img = img[rect[1]:rect[1]+rect[3], rect[0]: rect[0]+rect[2], :]

        # normalize
        sample = np.array(crop_img)
        sample = self.transform(sample)

        # rect.shape = [X, Y, W, H] (X, Y) - top left corner, return for the BB regresion
        # sampel.sape = [C, H, W]
        return sample, torch.tensor(label).long(), torch.tensor(rect).float()

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

''' ###################################### MAIN ###################################### 

Sample how to use the class                                                       
data_dir   = '/Users/royhirsch/Documents/GitHub/DetectionProject/ProcessedData'

Declaration of the class;
TrainDataLoader = BusDataLoader(data_dir)

Example of call to the iterator
img, label, rect = TrainDataLoader.__getitem__(650)

good examples: 2728, 650, 965, 2814, 634175, 634443
data_dir   = '/Users/royhirsch/Documents/GitHub/DetectionProject/ProcessedData'
TrainDataLoader = BusDataLoader(root_dir=data_dir, BGpad=16, outShape=225)
img, label, rect = TrainDataLoader.__getitem__(2728)
img, label, rect = TrainDataLoader.__getitem__(965)
img, label, rect = TrainDataLoader.__getitem__(2814)
img, label, rect = TrainDataLoader.__getitem__(650)
'''
