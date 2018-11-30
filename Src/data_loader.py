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
from utils import *

class BusDataLoader(Dataset):
    def __init__(self, root_dir, BGpad=16, outShape=225):
        """
        DataLoader class for handling and pre-processing of the data
        This class handles picked files containing pre-found region of interest rects that
        where found by selective search algorithm.

        Args:
            root_dir (string)                       : Path to the root folder of the pickeled files
            BGpad (int, must be equal)              : How much background padding to add
            outShape (int)                          : The desired input shape to the BB NN

        Methods:
        __getitem__ - generates a fixed size warped image to enter the NN backbone model and a GT label
        __len__     - number of data samples in the class

        """
        self.root_dir = root_dir
        self.BGpad = BGpad
        self.outShape = outShape

        self.rect_data = self._process_data()
        self.resized_images = self._process_images()

        # TODO: bug fix, the use of Resize flips the image
        self.transform  = transforms.Compose([
                          transforms.Resize(size=[self.outShape, self.outShape]),
                          transforms.ToTensor()])

    def _process_data(self, columns_list_rois=['image_name', 'rect', 'label']):
        data_frame = pd.DataFrame([], columns=columns_list_rois)

        for root, dirs, files in os.walk(self.root_dir):
            for name in files:
                if re.findall(r'rois_for_.*', name):
                    split_name = name.replace('.', '_').split('_')
                    rects = pickle.load(open(os.path.join(self.root_dir, name), "rb"))

                    temp = pd.DataFrame([[split_name[2], rect[:-1], rect[-1]] for rect in rects], columns=columns_list_rois)
                    data_frame = data_frame.append(temp, ignore_index=True)

        return data_frame

    def _process_images(self, columns_list=['image_name', 'image']):
        data_frame = pd.DataFrame([], columns=columns_list)
        i = 0

        for root, dirs, files in os.walk(self.root_dir):
            for name in files:
                if re.findall(r'resized_img_.*', name):
                    split_name = name.replace('.', '_').split('_')
                    img = pickle.load(open(os.path.join(self.root_dir, name), "rb"))

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
        sample = transforms.ToPILImage()(sample)
        sample = self.transform(sample)

        return torch.tensor(sample).float(), torch.tensor(label).long()

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

Example of call to the itterator
img, label = TrainDataLoader.__getitem__(650)

good examples: 2728, 650, 965, 2814, 634175, 634443

'''

