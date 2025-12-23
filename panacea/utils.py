"""
Utility Functions for Panacea

Image loading, saving, and quality metrics.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
import math


def load_image(path: Union[str, Path], size: int = 224) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load an image and prepare it for processing.
    
    Args:
        path: Path to the image file
        size: Target size for the image (default: 224 for CLIP)
        
    Returns:
        Tuple of (tensor of shape (1, 3, H, W) in range [0, 1], original PIL image)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Load original image
    original = Image.open(path).convert("RGB")
    
    # Resize while maintaining aspect ratio, then center crop
    transform = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),  # Converts to [0, 1] range
    ])
    
    tensor = transform(original).unsqueeze(0)  # Add batch dimension
    return tensor, original


def save_image(tensor: torch.Tensor, path: Union[str, Path], quality: int = 95):
    """
    Save a tensor as an image file.
    
    Args:
        tensor: Image tensor of shape (1, 3, H, W) or (3, H, W) in range [0, 1]
        path: Output path
        quality: JPEG quality (only for JPEG format)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Clamp and convert to PIL
    tensor = tensor.clamp(0, 1)
    pil_image = TF.to_pil_image(tensor.cpu())
    
    # Save with appropriate format
    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        pil_image.save(path, "JPEG", quality=quality)
    elif suffix == ".png":
        pil_image.save(path, "PNG")
    else:
        pil_image.save(path)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to PIL Image.
    
    Args:
        tensor: Image tensor of shape (1, 3, H, W) or (3, H, W) in range [0, 1]
        
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.clamp(0, 1)
    return TF.to_pil_image(tensor.cpu())


def pil_to_tensor(image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """
    Convert a PIL Image to tensor.
    
    Args:
        image: PIL Image
        device: Target device
        
    Returns:
        Tensor of shape (1, 3, H, W) in range [0, 1]
    """
    tensor = TF.to_tensor(image).unsqueeze(0)
    return tensor.to(device)


def compute_psnr(original: torch.Tensor, perturbed: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between original and perturbed images.
    Higher PSNR means less visible difference (30+ dB is typically imperceptible).
    
    Args:
        original: Original image tensor
        perturbed: Perturbed image tensor
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((original - perturbed) ** 2).item()
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def compute_linf_norm(original: torch.Tensor, perturbed: torch.Tensor) -> float:
    """
    Compute L-infinity norm (maximum pixel difference).
    
    Args:
        original: Original image tensor
        perturbed: Perturbed image tensor
        
    Returns:
        Maximum absolute difference
    """
    return (original - perturbed).abs().max().item()


def compute_l2_norm(original: torch.Tensor, perturbed: torch.Tensor) -> float:
    """
    Compute L2 norm (Euclidean distance).
    
    Args:
        original: Original image tensor
        perturbed: Perturbed image tensor
        
    Returns:
        L2 distance
    """
    return torch.sqrt(torch.sum((original - perturbed) ** 2)).item()
