"""
CLIP Model Wrapper for Panacea

Provides functions to load and use CLIP models for feature extraction.
"""

import torch
import open_clip
from typing import Tuple, List, Union
from PIL import Image


class CLIPWrapper:
    """Wrapper class for CLIP model operations."""
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None):
        """
        Initialize CLIP model wrapper.
        
        Args:
            model_name: CLIP model architecture (default: ViT-B-32)
            pretrained: Pretrained weights source (default: openai)
            device: Device to run model on (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Store normalization parameters for preprocessing
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
    
    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Apply CLIP normalization to images."""
        return (images - self.mean) / self.std
    
    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        """Remove CLIP normalization from images."""
        return images * self.std + self.mean
    
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized image features from CLIP.
        
        Args:
            images: Tensor of shape (B, C, H, W) in range [0, 1]
            
        Returns:
            Normalized image embeddings of shape (B, D)
        """
        # Apply CLIP normalization
        normalized = self.normalize(images)
        
        # Get features
        with torch.no_grad():
            features = self.model.encode_image(normalized)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
    def get_image_features_grad(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features with gradient computation enabled.
        
        Args:
            images: Tensor of shape (B, C, H, W) in range [0, 1]
            
        Returns:
            Normalized image embeddings of shape (B, D)
        """
        # Apply CLIP normalization
        normalized = self.normalize(images)
        
        # Get features with gradients
        features = self.model.encode_image(normalized)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
    def get_text_features(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Extract normalized text features from CLIP.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Normalized text embeddings of shape (B, D)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        tokens = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between image and text features.
        
        Args:
            image_features: Image embeddings of shape (B, D)
            text_features: Text embeddings of shape (N, D)
            
        Returns:
            Similarity matrix of shape (B, N)
        """
        return image_features @ text_features.T


def load_clip_model(model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None) -> CLIPWrapper:
    """
    Load a CLIP model.
    
    Args:
        model_name: CLIP model architecture
        pretrained: Pretrained weights source
        device: Device to run on
        
    Returns:
        CLIPWrapper instance
    """
    return CLIPWrapper(model_name, pretrained, device)
