import argparse
import cv2
import os
import torch

import neptune.new as neptune

from torch.utils.data import DataLoader

from trainer import Trainer
from utils import get_paths, get_train_transform, get_val_transform

from criteria.loss import LossModel
from models.mobile_net import MobileNet
from data.coco_dataset import CocoDataset


def main():
    """Train pipeline"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', type=str, default='data/',
                        help='Folder with train, val and styles data. Default: data/')
    parser.add_argument('--style', type=str, default='data/styles/wave.jpeg',
                        help='Style exemplar. Default: data/styles/wave.jpeg')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size. Default: 16')
    parser.add_argument('--lr', type=int, default=1e-3,
                        help='Learning rate value. Default: 1e-3')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs amount. Default: 30')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num of workers. Default: 4')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device. Default: cuda')
    parser.add_argument('--load_size', type=int, default=128,
                        help='Input image size. Default: 128')
    parser.add_argument('--neptune_project', type=str, default='ternerss/style-transfer',
                        help='Neptune project name. Default: ternerss/style-transfer')

    args = parser.parse_args()

    train_dir = os.path.join(args.dataroot, 'train')
    val_dir = os.path.join(args.dataroot, 'val')
    style = cv2.imread(args.style)[:, :, ::-1]

    train_transform = get_train_transform(args.load_size, args.load_size)
    val_transform = get_val_transform()

    train_ds = CocoDataset(get_paths(train_dir), train_transform)
    val_ds = CocoDataset(get_paths(val_dir), val_transform)

    train_dl = DataLoader(train_ds,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

    val_dl = DataLoader(val_ds,
                        batch_size=1,
                        num_workers=args.num_workers)

    transformed_style = val_transform(
        image=style)['image'].repeat(args.batch_size, 1, 1, 1)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = MobileNet().to(device)
    print(model)

    run = neptune.init(project=args.neptune_project,
                       api_token=os.environ.get('NEPTUNE_API_TOKEN'))

    criterion = LossModel(transformed_style, device, run)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_dl,
        val_loader=val_dl,
        epochs=args.epochs,
        run=run,
        device=device
    )

    model = trainer.start()


if __name__ == '__main__':
    main()
