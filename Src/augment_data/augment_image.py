import numpy as np
import pandas as pd
import cv2
# import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from data_aug import *
import imgaug as ia
from imgaug import augmenters as iaa
from torchvision import datasets, models, transforms

'''
    Simple pipline to yield transformed bus images
'''

class AugBusDataLoader(Dataset):
    def __init__(self, root_image_dir, label_path, scale=0.25):
        self.root_image_dir = root_image_dir
        self.label_path = label_path
        self.scale = scale

        self.gt_images = self._get_raw_images()
        self.gt_labels = self._get_gt_labels()

        self.seq = iaa.Sequential([iaa.Fliplr(0.5),
                              iaa.Flipud(0.2),
                              iaa.Sometimes(0.25, iaa.Multiply((1.2, 1.5))),
                              iaa.Affine(rotate=(-25, 25)),
                              iaa.Sometimes(0.1, iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})),
                              iaa.Sometimes(0.25, iaa.Affine(translate_px={"x": 40, "y": 60}))
                              ])

        self.transform = transforms.Compose([
                         transforms.ToPILImage(),
                         transforms.ToTensor()])

    def _get_raw_images(self):
        data_frame = pd.DataFrame([], columns=['image_name', 'image'])
        i = 0
        for root, dirs, files in os.walk(self.root_image_dir):
            for name in files:
                split_name = name.replace('.', '_').split('_')

                image_path = os.path.join(root, name)
                im = cv2.imread(image_path)

                # Resize image by scale
                newWidth = int(im.shape[0] * self.scale)
                newHeight = int(im.shape[1] * self.scale)
                im = cv2.resize(im, (newHeight, newWidth))

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
                    scale_roi = [int(num * self.scale) for num in roi[:-1]]
                    scale_roi.append(roi[-1])
                    scale_roi = conv_rect_to_opos_points(scale_roi)
                    scale_rois.append(scale_roi)

                # Good np format for the augmentation package
                np_rois = np.array(scale_rois)[:, :-1]
                labels_raw.loc[i] = [file_name, np_rois, scale_rois]

        return labels_raw

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, idx):
        im_name, img = self.gt_images.loc[idx]
        _, bboxes, rect = self.gt_labels.loc[self.gt_labels['image_name'] == im_name+'.JPG'].values[0]

        # Create ia.BoundingBox object list
        bbox_obj_ls = []
        for i in range(np.shape(bboxes)[0]):
            bbox_obj_ls.append(ia.BoundingBox(*bboxes[i]))

        # Create bbox object with img
        bbs = ia.BoundingBoxesOnImage(bbox_obj_ls, shape=img.shape)

        seq_det = self.seq.to_deterministic()

        # Apply augmentation
        image_aug = seq_det.augment_images([img])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        image = self.transform(image_aug)
        labels = torch.tensor(bbs_aug.to_xyxy_array()).long()

        # img_, bboxes_ = self.seq(img.copy(), np_rect.copy())

        return image, labels, bbs_aug

# From [xTopLeft,yTopLeft,w,h,c] to [xTopLeft,yTopLeft,xBottomRight,yBottomRight,c]
def conv_rect_to_opos_points(rect):
    if len(rect) == 4:
        return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
    else:
        return [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3], rect[4]]

def draw_img_augment(img, bbs):
    if type(img) == torch.Tensor:
        img = img.numpy()
        img = np.swapaxes(img, 0, -1)
        img = np.swapaxes(img, 0, 1)
    img_mod = bbs.draw_on_image(img.copy(), thickness=2)
    plt.figure()
    plt.imshow(img_mod)
'''
################ MAIN ################
images_dir = '/Users/royhirsch/Documents/Study/Current/ComputerVision/project/busesTrain'
gt_labels_path = '/Users/royhirsch/Documents/Study/Current/ComputerVision/project/annotationsTrain.txt'
aug_data = AugBusDataLoader(images_dir, gt_labels_path)
i, bb, bbt =  aug_data.__getitem__(12)
i, bb, bbt =  aug_data.__getitem__(12)
'''




