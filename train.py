import argparse
import cv2
import os
import torch


from trainer import Trainer
from utils import get_paths, get_train_transform, get_val_transform

from criteria.loss import LossModel
from models.mobile_net import TransformerMobileNet
from data.coco_dataset import CocoDataset

from torch.utils.data import DataLoader

import neptune.new as neptune


def main():
    """Train pipeline"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataroot', type=str, default='data/',
                        help='Folder with train, val and styles data. Default: data/')

    args = parser.parse_args()

    train_dir = os.path.join(args.dataroot, 'train')
    styles_dir = os.path.join(args.dataroot, 'styles')
    val_dir = os.path.join(args.dataroot, 'val')

    style = cv2.imread(os.path.join(styles_dir, 'wave.jpeg'))[:, :, ::-1]

    h, w = (128, 128)
    batch_size = 16
    lr = 1e-3
    num_workers = 4
    epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PROJECT = "ternerss/style-transfer"
    API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
    run = neptune.init(project=PROJECT, api_token=API_TOKEN)

    train_transform = get_train_transform(h, w)
    val_transform = get_val_transform()

    images_dir = get_paths(train_dir)

    train_ds = CocoDataset(images_dir, train_transform)
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

    transformed_style = val_transform(
        image=style)['image'].repeat(batch_size, 1, 1, 1)

    criterion = LossModel(transformed_style, device)
    model = TransformerMobileNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        loader=train_dl,
        epochs=epochs,
        check_image=os.path.join(val_dir, 'doge.jpg'),
        run=run,
        device=device
    )

    model = trainer.start()


if __name__ == '__main__':
    main()
