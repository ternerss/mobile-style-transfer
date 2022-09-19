import argparse
import cv2
import os
import torch
import numpy as np

from utils import get_inversed_image, get_val_transform

from models.mobile_net import MobileNet


def process_image(model, img):
    model.eval()

    with torch.no_grad():
        out_image = model(img.unsqueeze(0))
        out_image = out_image.squeeze(0).permute(1, 2, 0)
        out_image = out_image.detach().cpu().numpy()
        out_image = np.clip(get_inversed_image(out_image), 0, 1)

        return out_image


def main():
    """Test pipeline"""

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/val/doge.jpg',
                        help='Image path. Default: data/val/doge.jpg')
    parser.add_argument('--weights', type=str, default='weights/mobile_style_transfer.pth',
                        help='Model weights path. Default: weights/mobile_style_transfer.pth')
    parser.add_argument('--out', type=str, default='out/',
                        help='Output folder. Default: out/')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device. Default: cuda')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"{args.input}")

    img = cv2.imread(args.input)[:, :, ::-1]
    aug = get_val_transform()
    inp_img = aug(image=img)['image']

    model = MobileNet().to(device)
    model.load_state_dict(torch.load(args.weights))

    out_img = process_image(model, inp_img.to(device))
    out_img = np.uint(out_img * 255)[:, :, ::-1]

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    f, _ = os.path.splitext(os.path.basename(args.input))
    cv2.imwrite(f"{args.out}/stylized_{f}.png", out_img)


if __name__ == '__main__':
    main()
