import os
import os.path as osp

import cv2
import numpy as np
import random
from torch.utils import data
from PIL import Image
import torchvision.transforms as T
import torch

class PairedCityscapes(data.Dataset):
    def __init__(self, root_dir, set='train', max_iters=None, img_size=640):
        self.root_dir = root_dir
        self.set = set
        self.img_size = img_size
        # Define paths to CW and SF images and labels
        self.cw_image_dir = osp.join(root_dir, 'CW', 'images', set)
        self.sf_image_dir = osp.join(root_dir, 'SF', 'images', set)
        self.cw_label_dir = osp.join(root_dir, 'CW', 'labels', set)
        self.sf_label_dir = osp.join(root_dir, 'SF', 'labels', set)
        # List image files directly from CW image directory
        self.img_ids = [f for f in os.listdir(self.cw_image_dir) if f.endswith('.png')]

        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        for img_id in self.img_ids:
            self.files.append({
                'cw_img': osp.join(self.cw_image_dir, img_id),
                'sf_img': osp.join(self.sf_image_dir, img_id),
                'cw_label': osp.join(self.cw_label_dir, img_id.replace('.png', '.txt')),
                'sf_label': osp.join(self.sf_label_dir, img_id.replace('.png', '.txt')),
                'name': img_id
            })
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        # Load and preprocess images
        cw_img = Image.open(datafiles['cw_img']).convert('RGB')
        sf_img = Image.open(datafiles['sf_img']).convert('RGB')

        # remember original sizes (width, height)
        cw_w, cw_h = cw_img.size
        sf_w, sf_h = sf_img.size

        # Letterbox each image -> returns PIL image, scale ratio, (pad_left, pad_top)
        cw_img, cw_r, (cw_pad_left, cw_pad_top) = self.letterbox(cw_img, self.img_size)
        sf_img, sf_r, (sf_pad_left, sf_pad_top) = self.letterbox(sf_img, self.img_size)

        # Convert to tensor
        cw_img = self.transform(cw_img)
        sf_img = self.transform(sf_img)

        # Load and preprocess labels
        cw_label = self.load_yolo_label(datafiles['cw_label'])
        sf_label = self.load_yolo_label(datafiles['sf_label'])

        # Resize label boxes to match letterboxed image (normalized w.r.t final img_size)
        cw_label = self.resize_yolo_labels(cw_label, cw_w, cw_h, cw_r, (cw_pad_left, cw_pad_top), self.img_size)
        sf_label = self.resize_yolo_labels(sf_label, sf_w, sf_h, sf_r, (sf_pad_left, sf_pad_top), self.img_size)

        return cw_img, sf_img, cw_label, sf_label, datafiles['name'], 'CW', 'SF'

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

    def load_yolo_label(self, label_path):
        boxes, labels = [], []
        if osp.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    boxes.append([x, y, w, h])
                    labels.append(int(class_id))
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

    def resize_yolo_labels(self, label_dict, orig_w, orig_h, scale_r, pad, final_size):
        """
        Convert YOLO normalized boxes from original image -> letterboxed image, and re-normalize
        relative to final_size (e.g. 640).
        - label_dict: {'boxes': tensor[N,4] (x,y,w,h normalized), 'labels': tensor[N]}
        - orig_w, orig_h: original image width & height (ints)
        - scale_r: scale applied to original image (float)
        - pad: (pad_left, pad_top) in pixels
        - final_size: int final (square) size
        """
        boxes = label_dict['boxes']
        labels = label_dict['labels']

        # empty case
        if boxes.numel() == 0:
            return {'boxes': torch.zeros((0, 4), dtype=torch.float32), 'labels': torch.zeros((0,), dtype=torch.int64)}

        # ensure shape Nx4
        if boxes.dim() == 1:
            boxes = boxes.view(1, 4)

        # convert normalized -> pixel coords in original
        x_c = boxes[:, 0] * orig_w
        y_c = boxes[:, 1] * orig_h
        bw = boxes[:, 2] * orig_w
        bh = boxes[:, 3] * orig_h

        # apply scaling and padding
        pad_left, pad_top = pad
        x_c = x_c * scale_r + pad_left
        y_c = y_c * scale_r + pad_top
        bw = bw * scale_r
        bh = bh * scale_r

        # normalize back relative to final_size
        x_c = x_c / float(final_size)
        y_c = y_c / float(final_size)
        bw = bw / float(final_size)
        bh = bh / float(final_size)

        new_boxes = torch.stack([x_c, y_c, bw, bh], dim=1).type(torch.float32)
        return {'boxes': new_boxes, 'labels': labels}

    def collate_fn(self, batch):
        cw_imgs, sf_imgs, cw_labels, sf_labels, names, cw_domains, sf_domains = zip(*batch)

        # Stack image tensors
        cw_imgs = torch.stack(cw_imgs, dim=0)
        sf_imgs = torch.stack(sf_imgs, dim=0)

        # Keep labels as lists of dicts
        return cw_imgs, sf_imgs, list(cw_labels), list(sf_labels), list(names), list(cw_domains), list(sf_domains)