import torch
import os
import numpy as np

from collections import defaultdict

from neptune.new.types import File
from tqdm import tqdm

from utils import get_inversed_image


class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, epochs, device, run):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs
        self.device = device

        self.run = run

    def _train(self):
        self.model.train()

        results = defaultdict(list[float])

        for images in tqdm(self.train_loader):
            self.optimizer.zero_grad()

            images = images.to(self.device)
            out = self.model(images)

            losses = self.criterion(out, images)
            losses["mixed_loss"].backward()
            self.optimizer.step()

            for name, value in losses.items():
                results[name].append(value.item())

        return {name: np.mean(values) for name, values in results.items()}

    def _val(self):
        self.model.eval()

        with torch.no_grad():
            for idx, img in tqdm(enumerate(self.val_loader)):
                out_image = self.model(img.to(self.device))

                out_image = out_image.squeeze(0).permute(1, 2, 0)
                out_image = out_image.detach().cpu().numpy()
                out_image = np.clip(get_inversed_image(out_image), 0, 1)

                self.run[f"eval/image_{idx}"].upload(File.as_image(out_image))

    def start(self):
        for epoch in range(self.epochs):
            print(f"epoch: {epoch}")

            results = self._train()
            self._val()

            for name, value in results.items():
                print(f"{name}: {value}")
                self.run[f"train/{name}"].log(value)

            torch.save(self.model.state_dict(),
                       f"logs/mobile_style_transfer_{epoch}.pth")

        return self.model
