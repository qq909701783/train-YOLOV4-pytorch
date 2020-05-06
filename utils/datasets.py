import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import os.path as osp
from torch.utils.data import Dataset
import torchvision.transforms as transforms
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import cv2
from utils.augmentation import Resizer,Normalizer,Augmenter,collater
from torch.utils.data import DataLoader

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


def resizer(image,common_size=608):
    height, width, _ = image.shape
    if height > width:
        scale = common_size / height
        resized_height = common_size
        resized_width = int(width * scale)
    else:
        scale = common_size / width
        resized_height = int(height * scale)
        resized_width = common_size

    image = cv2.resize(image, (resized_width, resized_height))

    new_image = np.zeros((common_size, common_size, 3))
    new_image[0:resized_height, 0:resized_width] = image

    new_image = torch.from_numpy(new_image)

    return new_image,scale


def Normalizer(image):
   mean = np.array([[[0.485, 0.456, 0.406]]])
   std = np.array([[[0.229, 0.224, 0.225]]])
   image = (image.astype(np.float32) - mean)/std
   return image


# class ImageFolder(Dataset):
#     def __init__(self, folder_path, img_size=416):
#         self.files = sorted(glob.glob("%s/*.*" % folder_path))
#         self.img_size = img_size
#
#     def __getitem__(self, index):
#         img_path = self.files[index % len(self.files)]
#         # Extract image as PyTorch tensor
#         img = transforms.ToTensor()(Image.open(img_path))
#         # Pad to square resolution
#         img, _ = pad_to_square(img, 0)
#         # Resize
#         img = resize(img, self.img_size)
#         return img_path, img
#
#     def __len__(self):
#         return len(self.files)



class VOCAnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, scale):
        max_boxes = 50
        label = np.zeros((max_boxes,5))
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            cenbox = []
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)

            cenbox.append(bndbox[4])
            cenbox.append((bndbox[0]+bndbox[2])/2)
            cenbox.append((bndbox[1]+bndbox[3])/2)
            cenbox.append((bndbox[2]-bndbox[0]))
            cenbox.append((bndbox[3]-bndbox[1]))
            res += [cenbox]
        res = np.array(res)
        res[:,1:] *= scale

        cc = 0
        for i in range(res.shape[0]):
            label[cc] = res[i]
            cc += 1
            if cc >= 50:
                break
        label = np.reshape(label, (-1))
        label = np.expand_dims(label,axis=0)

        return label

class VOCDetection(Dataset):
    def __init__(self, root,
                 image_sets='train',
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712',tname = 'trainval'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()

        print(self.root)
        for line in open(osp.join(self.root, 'ImageSets', 'Main', image_sets + '.txt')):
            self.ids.append((self.root, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]


        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        img = Normalizer(img)
        img,scale = resizer(img)

        if self.target_transform is not None:
            boxes = self.target_transform(target, scale)
        # target = np.zeros((len(boxes),6))
        boxes = np.array(boxes)
        boxes = torch.from_numpy(boxes)
        # target[:, 1:] = boxes
        sample = {'img': img, 'annot': boxes}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.ids)

    def num_classes(self):
        return len(VOC_CLASSES)

    def label_to_name(self, label):
        return VOC_CLASSES[label]

    def load_annotations(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        gt = np.array(gt)
        return gt

if __name__ == '__main__':
    dataset_root = r'/disk_d/workspace/personalSpace/like_project/VOC_/VOCdevkit/VOC2007'

    train_dataset = VOCDetection(root=dataset_root)
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              num_workers=0,
                              shuffle=True,
                              collate_fn=collater,
                              pin_memory=True)
    for idx, (images,targets) in enumerate(train_loader):
        print(targets)
        break