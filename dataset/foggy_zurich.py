import os
import os.path as osp

import cv2
import numpy as np
import random
from torch.utils import data
from PIL import Image
import torchvision.transforms as T
import torch

class FoggyZurich(data.Dataset):
    def __init__(self, root_dir, set='train', max_iters=None, img_size=640):
        self.root_dir = root_dir
        self.set = set
        self.img_size = img_size
        self.rf_image_dir = osp.join(root_dir, 'RF', 'images', set)
        self.img_ids = [f for f in os.listdir(self.rf_image_dir) if f.endswith('.png')]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for img_id in self.img_ids:
            self.files.append({
                'rf_img': osp.join(self.rf_image_dir, img_id),
                'name': img_id
            })
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        rf_img = Image.open(datafiles['rf_img']).convert('RGB')
        rf_img, rf_r, (rf_pad_left, rf_pad_top) = self.letterbox(rf_img, self.img_size)
        rf_img = self.transform(rf_img)
        return rf_img, datafiles['name'], 'RF'

    def letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        """
        img: PIL.Image (RGB)
        new_shape: int or (h, w) target. We'll use square int for compatibility with YOLO.
        returns: (PIL.Image padded), scale_r, (pad_left, pad_top)
        """
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        img_np = np.array(img)  # H x W x C, RGB
        h0, w0 = img_np.shape[:2]

        # scale ratio
        r = min(new_shape[0] / h0, new_shape[1] / w0)
        new_unpad_w = int(round(w0 * r))
        new_unpad_h = int(round(h0 * r))

        # resize
        img_resized = cv2.resize(img_np, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

        # compute padding
        dw = new_shape[1] - new_unpad_w
        dh = new_shape[0] - new_unpad_h
        left = int(np.floor(dw / 2))
        right = int(dw - left)
        top = int(np.floor(dh / 2))
        bottom = int(dh - top)

        # pad
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        # convert back to PIL (still RGB)
        img_pil = Image.fromarray(img_padded)

        return img_pil, r, (left, top)

    def collate_fn(self, batch):
        """
        Hàm này được sử dụng để gom các mẫu trong batch lại với nhau.
        """
        fog_images, img_names, domains = zip(*batch)
        fog_images = torch.stack(fog_images, 0)
        return fog_images, img_names, list(domains)