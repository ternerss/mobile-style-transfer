import cv2

from tqdm import tqdm
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, image_dir, train_transform=None):
        self.images = [cv2.imread(f)[:, :, ::-1] for f in tqdm(image_dir)]
        self.aug = train_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        sample = self.aug(image=image) if self.aug else {
            'image': image,
        }

        return sample['image']
