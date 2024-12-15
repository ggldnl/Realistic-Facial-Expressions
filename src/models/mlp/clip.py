from typing import Optional

import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIP(pl.LightningModule):
    """Wrapper for CLIP model and related transforms using Hugging Face implementation"""

    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 res: int = 224):
        super().__init__()
        self.res = res
        self.save_hyperparameters()

        # Load CLIP model and processor from Hugging Face
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # CLIP normalization values from original implementation
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
        inputs = self.processor(text=[prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_features = self.model.get_text_features(**inputs)
        return text_features

    def encode_image(self, image_path: str) -> torch.Tensor:
        """Encode reference image using CLIP"""
        img = Image.open(image_path)
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_features = self.model.get_image_features(**inputs)
        return image_features

    def encode_renders(self, renders: torch.Tensor) -> torch.Tensor:
        """Encode rendered views using CLIP"""
        processed = self.transform(renders)
        inputs = {"pixel_values": processed.to(self.device)}
        image_features = self.model.get_image_features(**inputs)
        return image_features

    def encode_augmented_renders(self, renders: torch.Tensor) -> torch.Tensor:
        """Encode rendered views with augmentation using CLIP"""
        processed = self.augment_transform(renders)
        inputs = {"pixel_values": processed.to(self.device)}
        image_features = self.model.get_image_features(**inputs)
        return image_features