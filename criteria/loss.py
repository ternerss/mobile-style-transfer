import numpy as np
import torch.nn.functional as F

from torch import nn
from torchvision.models import vgg16

from .modules import GradLayer
from .extractor import FeatureExtractor


class LossModel(nn.Module):
    def _get_gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, w*h)
        features_t = features.transpose(1, 2)
        return features.bmm(features_t) / (c * h * w)

    def _get_gramms(self, feautres):
        return [self._get_gram_matrix(feature) for feature in feautres]

    def __init__(
        self,
        style,
        device,
        style_weight=1,
        content_weight=1,
        sobel_weight=1,
        content_idx=1,
        layers_proportions=[.35, .35, .15, .15],
        layers_weight=3e5,
    ):
        super().__init__()


        self.style_weight = style_weight
        self.content_weight = content_weight
        self.sobel_weight = sobel_weight

        self.content_idx = content_idx
        self.layers_impacts = np.array(layers_proportions) * layers_weight

        self.extractor = FeatureExtractor(model=vgg16(pretrained=True), layer_idx=[3, 8, 15, 22]).to(device)
        style_features = self.extractor(style.to(device))
        self.style_gramms = self._get_gramms(style_features)

        self.device = device

    def _get_style_loss(self, generated):
        generated_gramms = self._get_gramms(generated)

        # to numpy
        loss = 0
        for x_gramm, y_gramm, w in zip(generated_gramms, self.style_gramms, self.layers_impacts):
            loss += w * F.mse_loss(x_gramm, y_gramm)

        return loss

    def _get_content_loss(self, generated_features, content_features):
        return F.mse_loss(generated_features[self.content_idx], content_features[self.content_idx])

    def _get_sobel_loss(self, generated, content):
        loss = nn.L1Loss()
        grad_layer = GradLayer().to(self.device)

        output_grad = grad_layer(generated)
        gt_grad = grad_layer(content)

        return loss(output_grad, gt_grad)

    def forward(self, generated, content):
        generated_features = self.extractor(generated)
        content_features = self.extractor(content)

        content_loss = self._get_content_loss(
            generated_features, content_features)
        style_loss = self._get_style_loss(generated_features)
        sobel_loss = self._get_sobel_loss(generated, content)

        mixed_loss = self.style_weight * style_loss + self.content_weight * \
            content_loss + self.sobel_weight * sobel_loss

        return mixed_loss, style_loss, content_loss, sobel_loss
