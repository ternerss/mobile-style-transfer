
import os
import numpy as np

import albumentations as A
import albumentations.pytorch.transforms as tr

VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD = (0.229, 0.224, 0.225)

def get_paths(path):
    _, _, filenames = next(os.walk(path))

    images_paths = []
    for filename in sorted(filenames):
        images_paths.append(os.path.join(path, filename))

    return np.stack(images_paths)


def get_train_transform(h, w, channel_mean=VGG_MEAN, channel_std=VGG_STD):
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(channel_mean, channel_std),
        tr.ToTensorV2(),
    ])


def get_val_transform(channel_mean=VGG_MEAN, channel_std=VGG_STD):
    return A.Compose([
        A.Normalize(channel_mean, channel_std),
        tr.ToTensorV2(),
    ])

def get_inversed_image(img):
    return img * VGG_STD + VGG_MEAN