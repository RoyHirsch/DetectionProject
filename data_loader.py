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

class BusDataLoader(Dataset):
    def __init__(self, root_dir, label_path, transform=None):
        """
        DataLoader class for handling and pre-processing of the data
        Args:
            root_dir (string)       : Path to the root folder of the images
                                      (make sure only images inside)
            label_path (string)     : Path to the images txt GT file

        """
        self.root_dir = root_dir
        self.label_path = label_path
        self.transform = transform

        self.labels = self._process_labels_file()
        self.raw_data, self.rois_data = self._process_data()

        self.data, self.labels = self._arrange_data_and_labels()

    def _process_labels_file(self, columns_list=['image_name', 'rois']):
        data_frame = pd.DataFrame([], columns=columns_list)
        with open(self.label_path) as fp:
            for i, line in enumerate(fp):
                tmp = line.split(':')
                file_name = tmp[0]
                rois = list(tmp[1].replace('],[', ' ').replace('[', ' ').replace(']', ' ').split())
                rois = [list(map(int, item.split(','))) for item in rois]
                data_frame.loc[i] = [file_name, rois]
        return data_frame

    def _process_data(self, scale=0.25, columns_list_raw_data=['image_name', 'raw_image'],
                      columns_list_rois=['image_name', 'rois']):
        raw_data = pd.DataFrame([], columns=columns_list_raw_data)
        rois_data = pd.DataFrame([], columns=columns_list_rois)

        print('Start processing the data')
        for root, dirs, files in os.walk(self.root_dir):
            for i, name in enumerate(files):
                image_path = os.path.join(root, name)
                im = cv2.imread(image_path)

                # Resize im by scale
                newWidth = int(im.shape[0] * scale)
                newHeight = int(im.shape[1] * scale)
                im = cv2.resize(im, (newWidth, newHeight))
                raw_data.loc[i] = [name, im]

                rois = selective_search(im)
                # Document the rois
                # for j, roi in enumerate(rois):
                    # rois_data.loc[j] = [name, roi]
                print('Ended to process sample number {}'.format(i))

        return raw_data, rois_data

    def _arrange_data_and_labels(self):
        merged = pd.merge(self.data, self.labels, on='image_name')
        merged.sort_values('image_name').reset_index(inplace=True)
        return merged.iloc[:, 1].values, merged.iloc[:, 2].values

    def __len__(self):
        return len(self.data)

    # Getter, normalizes the data and the labels
    def __getitem__(self, idx):
        sample = Image.open(self.data[idx])
        # sample = np.asarray(sample)
        sample_labels = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        return torch.tensor(sample).float(), torch.tensor(sample_labels).float()

def selective_search(im, printRect=False, method='reg'):

    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # im = cv2.imread(imgPath)
    # resize image
    # newHeight = 200
    # newWidth = int(im.shape[1] * 200 / im.shape[0])
    # im = cv2.resize(im, (newWidth, newHeight))

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

    if printRect:
        # number of region proposals to show
        numShowRects = 100

        # create a copy of original image
        imOut = im.copy()
        # itereate over all the region proposals
        for i in range(numShowRects):
            # draw rectangle for region proposal till numShowRects
            x, y, w, h = rects[i]
            cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

        # show output
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(imOut)
        plt.show()

    return rects

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


# get_mean_and_std(datasets.ImageFolder(root=<path_to_root>, transform=transforms.ToTensor()))
# selective_search('/Users/royhirsch/Documents/Study/Current/ComputerVision/project/busesTrain/DSCF1015.JPG')
data_transform = transforms.Compose([
                 transforms.Resize(size=[256,256]),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5686, 0.5184, 0.4736],
                                      std=[0.2385, 0.2251, 0.2193])
                 ])

TrainDataLoader = BusDataLoader(root_dir='/Users/royhirsch/Documents/Study/Current/ComputerVision/project/busesTrain',
                                label_path='/Users/royhirsch/Documents/Study/Current/ComputerVision/project/annotationsTrain.txt',
                                transform=data_transform)
TrainDataLoader.__getitem__(1)
