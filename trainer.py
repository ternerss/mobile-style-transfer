import torch
import cv2
import numpy as np
import neptune.new as neptune

from neptune.new.types import File
from tqdm import tqdm

from utils import get_val_transform, get_inversed_image

class Trainer:
    def __init__(self, model, optimizer, criterion, loader, epochs, check_image, device, run):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.loader = loader
        self.epochs = epochs

        self.val_transform = get_val_transform()

        self.check_image = check_image
        self.device = device

        self.run = run
    
    def _train(self):
        self.model.train()

        losses = []
        style_losses = []
        content_losses = []
        sobel_losses = []
        
        for images in tqdm(self.loader):
            self.optimizer.zero_grad()

            images = images.to(self.device)
            out = self.model(images)
        
            mixed_loss, style_loss, content_loss, sobel_loss = self.criterion(out, images)
            mixed_loss.backward()
            self.optimizer.step()

            losses.append(mixed_loss.item())
            style_losses.append(style_loss.item()) 
            content_losses.append(content_loss.item())
            sobel_losses.append(sobel_loss.item()) 
        
        return np.mean(losses), np.mean(style_losses), np.mean(content_losses), np.mean(sobel_losses)

    def _val(self):
        self.model.eval()

        with torch.no_grad():
            image = cv2.imread(self.check_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            transformed = self.val_transform(image=image)
            transformed_image = transformed['image']

            out_image = self.model(transformed_image.unsqueeze(0).to(self.device))
            out_image = out_image.squeeze(0).permute(1, 2, 0)
            out_image = out_image.detach().cpu().numpy()

        return get_inversed_image(out_image)

    def start(self):   
        self.run["params/style_weight"].log(self.criterion.style_weight)
        self.run["params/content_weight"].log(self.criterion.content_weight)
        self.run["params/content_idx"].log(self.criterion.content_idx)
        
        self.run["params/layers_impacts"].log(" ".join(str(x) for x in self.criterion.layers_impacts.tolist()))
        self.run["params/layer_idx"].log(" ".join(str(x) for x in self.criterion.extractor.layer_idx))
        
        for epoch in range(1, self.epochs+1):
            torch.cuda.empty_cache()

            mixed_loss, style_loss, content_loss, sobel_loss = self._train()
            result = np.clip(self._val(), 0, 1)
            print(f'mixed loss: {mixed_loss}, style loss: {style_loss}, content loss: {content_loss}, sobel loss: {sobel_loss}')
            
            self.run["train/mixed_loss"].log(mixed_loss)
            self.run["train/style_loss"].log(style_loss)
            self.run["train/content_loss"].log(content_loss)
            self.run["train/sobel_loss"].log(sobel_loss)
            self.run[f"eval/image_{epoch}"].upload(File.as_image(result))
        
        self.run.stop()

        return self.model