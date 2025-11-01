import torch
import torch.nn as nn
from torch.hub import load

# from aim.v2.utils import load_pretrained

dino_backbones = {
    "small": {"name": "dinov2_vits14_reg", "embedding_size": 384, "patch_size": 14},
    "base": {"name": "dinov2_vitb14_reg", "embedding_size": 768, "patch_size": 14},
    "large": {"name": "dinov2_vitl14_reg", "embedding_size": 1024, "patch_size": 14},
    "giant": {"name": "dinov2_vitg14_reg", "embedding_size": 1536, "patch_size": 14},
}


class DinoV2Patches(nn.Module):
    def __init__(self, in_chanels=3, out_channels=768, size="base"):
        from torchvision import transforms

        super(DinoV2Patches, self).__init__()
        self.size = size
        self.backbone = load("facebookresearch/dinov2", dino_backbones[self.size]["name"], pretrained=True)
        self.backbone.eval()
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self.out_channels = out_channels
        self.inet_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def transform(self, x):
        # x should have shape (B, C, H, W)
        b, c, h, w = x.shape

        # Compute how many pixels to drop to make height/width multiples of 14
        h_new = h - (h % 14)
        w_new = w - (w % 14)

        dh = h - h_new  # total pixels to drop in height
        dw = w - w_new  # total pixels to drop in width

        # Decide how many pixels to drop from each side.
        # For simplicity, drop half from each edge if dh/dw are even.
        # If dh or dw are odd, we'll drop one extra pixel from the "bottom/right".
        dh_top = dh // 2
        dh_bottom = dh - dh_top
        dw_left = dw // 2
        dw_right = dw - dw_left

        # Crop the tensor from edges
        # Make sure slicing indices do not go negative
        x_cropped = x[:, :, dh_top : h - dh_bottom, dw_left : w - dw_right]

        # normalize with imagenet mean and std
        x_cropped = self.inet_norm(x_cropped)

        return x_cropped

    def forward(self, x):
        with torch.no_grad():
            x = self.transform(x)
            batch_size = x.shape[0]
            mask_dim = (x.shape[2] / 14, x.shape[3] / 14)
            with torch.no_grad():
                x = self.backbone.forward_features(x)
                x = x["x_norm_patchtokens"]
                x = x.permute(0, 2, 1)
                x = x.reshape(batch_size, self.out_channels, int(mask_dim[0]), int(mask_dim[1]))
            return x
