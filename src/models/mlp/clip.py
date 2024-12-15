import clip
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from PIL import Image


class CLIP(pl.LightningModule):
    """Wrapper for CLIP model and related transforms"""

    def __init__(self,
                 model_name: str = "ViT-B/32",
                 res: int = 224,
                 device: str = "cuda"):
        super().__init__()
        self.device = device
        self.res = res

        # Load CLIP model
        self.model, _ = clip.load(model_name, device=device)

        # CLIP normalization
        self.normalizer = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )

        # Set up transforms
        self.transform = transforms.Compose([
            transforms.Resize((res, res)),
            self.normalizer
        ])

        self.augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(res, scale=(0.8, 1.0)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            self.normalizer
        ])

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt using CLIP"""
        token = clip.tokenize([prompt]).to(self.device)
        return self.model.encode_text(token)

    def encode_image(self, image_path: str) -> torch.Tensor:
        """Encode reference image using CLIP"""
        img = Image.open(image_path)
        img = self.transform(img).unsqueeze(0).to(self.device)
        return self.model.encode_image(img)

    def encode_renders(self, renders: torch.Tensor) -> torch.Tensor:
        """Encode rendered views using CLIP"""
        processed = self.transform(renders)
        return self.model.encode_image(processed)

    def encode_augmented_renders(self, renders: torch.Tensor) -> torch.Tensor:
        """Encode rendered views with augmentation using CLIP"""
        processed = self.augment_transform(renders)
        return self.model.encode_image(processed)
