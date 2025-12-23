"""
Perceptual Loss Module for Panacea

Provides LPIPS-based perceptual loss to ensure perturbations remain
invisible to humans while being effective against AI.
"""

import torch
import torch.nn as nn
import lpips
from typing import Optional


class PerceptualLoss:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) based loss.
    
    Uses a pretrained VGG network to measure perceptual similarity.
    Lower values = more perceptually similar (less visible perturbation).
    """
    
    def __init__(self, net: str = "vgg", device: str = None):
        """
        Initialize perceptual loss.
        
        Args:
            net: Network to use ('vgg', 'alex', 'squeeze')
            device: Device to run on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = lpips.LPIPS(net=net, verbose=False).to(self.device)
        self.loss_fn.eval()
        
        # Freeze the network
        for param in self.loss_fn.parameters():
            param.requires_grad = False
    
    def __call__(self, original: torch.Tensor, perturbed: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between original and perturbed images.
        
        Args:
            original: Original image tensor (B, C, H, W) in range [0, 1]
            perturbed: Perturbed image tensor (B, C, H, W) in range [0, 1]
            
        Returns:
            Perceptual loss value (scalar tensor)
        """
        # LPIPS expects images in range [-1, 1]
        orig_scaled = original * 2 - 1
        pert_scaled = perturbed * 2 - 1
        
        return self.loss_fn(orig_scaled, pert_scaled).mean()


class FrequencyMask:
    """
    Frequency-domain masking to hide perturbations in high-frequency regions.
    
    High-frequency perturbations are less visible to humans.
    """
    
    def __init__(self, size: int = 224, low_freq_ratio: float = 0.1, device: str = None):
        """
        Initialize frequency mask.
        
        Args:
            size: Image size (assumes square)
            low_freq_ratio: Ratio of frequencies to consider as "low" (0-1)
            device: Device to run on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        
        # Create frequency mask (higher weight for high frequencies)
        center = size // 2
        y, x = torch.meshgrid(
            torch.arange(size, device=self.device) - center,
            torch.arange(size, device=self.device) - center,
            indexing='ij'
        )
        distance = torch.sqrt(x**2 + y**2)
        max_dist = center * 1.414  # Diagonal
        
        # Low frequencies get low weight, high frequencies get high weight
        low_freq_radius = max_dist * low_freq_ratio
        self.mask = torch.clamp((distance - low_freq_radius) / max_dist, 0, 1)
        self.mask = self.mask.view(1, 1, size, size)
    
    def apply(self, perturbation: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency mask to perturbation.
        
        Reduces perturbation magnitude in low-frequency regions.
        
        Args:
            perturbation: Perturbation tensor (B, C, H, W)
            
        Returns:
            Masked perturbation
        """
        # FFT
        fft = torch.fft.fft2(perturbation)
        fft_shifted = torch.fft.fftshift(fft)
        
        # Apply mask
        fft_masked = fft_shifted * self.mask
        
        # Inverse FFT
        fft_unshifted = torch.fft.ifftshift(fft_masked)
        result = torch.fft.ifft2(fft_unshifted).real
        
        return result


class SaliencyMask:
    """
    Saliency-based masking to hide perturbations in low-attention regions.
    
    Perturbations in regions humans don't focus on are less noticeable.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize saliency mask.
        
        Args:
            device: Device to run on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_edge_mask(self, image: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """
        Compute edge-based saliency mask.
        
        Edges are salient - perturbations should avoid them.
        
        Args:
            image: Image tensor (B, C, H, W) in range [0, 1]
            threshold: Edge detection threshold
            
        Returns:
            Inverse saliency mask (high values = safe to perturb)
        """
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Convert to grayscale
        gray = image.mean(dim=1, keepdim=True)
        
        # Compute gradients
        pad = nn.functional.pad(gray, (1, 1, 1, 1), mode='replicate')
        gx = nn.functional.conv2d(pad, sobel_x)
        gy = nn.functional.conv2d(pad, sobel_y)
        
        # Edge magnitude
        edges = torch.sqrt(gx**2 + gy**2)
        edges = edges / edges.max()
        
        # Invert: high values where edges are NOT (safe to perturb)
        mask = 1 - torch.clamp(edges / threshold, 0, 1)
        
        # Expand to all channels
        mask = mask.expand(-1, image.shape[1], -1, -1)
        
        return mask
