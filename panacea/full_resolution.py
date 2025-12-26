"""
Full Resolution Processing for Panacea

Processes images at their original resolution by tiling and combining perturbations.
This preserves image fidelity while still applying effective attacks.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import math


class FullResolutionProcessor:
    """
    Process images at full resolution using tiled attacks.
    
    CLIP requires 224x224 input, but we can:
    1. Process overlapping tiles at 224x224
    2. Blend perturbations from tiles
    3. Apply combined perturbation to full-res image
    """
    
    def __init__(
        self,
        tile_size: int = 224,
        overlap: int = 32,
        blend_mode: str = "linear",
        device: str = None
    ):
        """
        Initialize full resolution processor.
        
        Args:
            tile_size: Size of each tile (should match model input size)
            overlap: Overlap between tiles for smooth blending
            blend_mode: How to blend overlapping tiles ('linear', 'max', 'average')
            device: Device to process on
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.blend_mode = blend_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stride = tile_size - overlap
    
    def extract_tiles(self, image: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Extract overlapping tiles from a full-resolution image.
        
        Args:
            image: Full resolution image (1, C, H, W)
            
        Returns:
            Tuple of (tiles tensor (N, C, tile_size, tile_size), tile positions)
        """
        _, C, H, W = image.shape
        
        tiles = []
        positions = []
        
        # Calculate number of tiles needed
        n_rows = max(1, math.ceil((H - self.overlap) / self.stride))
        n_cols = max(1, math.ceil((W - self.overlap) / self.stride))
        
        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate tile position
                y = min(row * self.stride, max(0, H - self.tile_size))
                x = min(col * self.stride, max(0, W - self.tile_size))
                
                # Extract tile
                tile = image[:, :, y:y+self.tile_size, x:x+self.tile_size]
                
                # Pad if necessary (for small images)
                if tile.shape[2] < self.tile_size or tile.shape[3] < self.tile_size:
                    pad_h = self.tile_size - tile.shape[2]
                    pad_w = self.tile_size - tile.shape[3]
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                
                tiles.append(tile)
                positions.append((y, x, min(y + self.tile_size, H), min(x + self.tile_size, W)))
        
        return torch.cat(tiles, dim=0), positions
    
    def combine_tiles(
        self,
        tiles: torch.Tensor,
        positions: list,
        original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Combine perturbed tiles back into full-resolution image.
        
        Args:
            tiles: Perturbed tiles (N, C, tile_size, tile_size)
            positions: List of (y1, x1, y2, x2) for each tile
            original_size: (H, W) of original image
            
        Returns:
            Combined perturbation (1, C, H, W)
        """
        H, W = original_size
        C = tiles.shape[1]
        
        # Accumulator for perturbations and weights
        combined = torch.zeros(1, C, H, W, device=tiles.device)
        weights = torch.zeros(1, 1, H, W, device=tiles.device)
        
        for i, (y1, x1, y2, x2) in enumerate(positions):
            tile = tiles[i:i+1]
            tile_h, tile_w = y2 - y1, x2 - x1
            
            # Create blending weight (linear ramp at edges)
            weight = self._create_blend_weight(tile_h, tile_w, tiles.device)
            
            # Add tile contribution
            combined[:, :, y1:y2, x1:x2] += tile[:, :, :tile_h, :tile_w] * weight
            weights[:, :, y1:y2, x1:x2] += weight
        
        # Normalize by weights
        combined = combined / (weights + 1e-8)
        
        return combined
    
    def _create_blend_weight(self, h: int, w: int, device) -> torch.Tensor:
        """Create linear blending weight for a tile."""
        if self.blend_mode == "linear":
            # Create linear ramps at edges
            ramp_size = self.overlap // 2
            
            weight_h = torch.ones(h, device=device)
            weight_w = torch.ones(w, device=device)
            
            if h > 2 * ramp_size:
                ramp = torch.linspace(0, 1, ramp_size, device=device)
                weight_h[:ramp_size] = ramp
                weight_h[-ramp_size:] = ramp.flip(0)
            
            if w > 2 * ramp_size:
                ramp = torch.linspace(0, 1, ramp_size, device=device)
                weight_w[:ramp_size] = ramp
                weight_w[-ramp_size:] = ramp.flip(0)
            
            weight = weight_h.view(-1, 1) * weight_w.view(1, -1)
            return weight.view(1, 1, h, w)
        else:
            return torch.ones(1, 1, h, w, device=device)


def load_image_full_res(
    path: Union[str, Path],
    max_size: Optional[int] = None,
    device: str = "cpu"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
    """
    Load an image at full resolution (or optionally limit max dimension).
    
    Args:
        path: Path to the image file
        max_size: Optional maximum dimension (preserves aspect ratio)
        device: Device to load tensor on
        
    Returns:
        Tuple of (tensor (1, 3, H, W), original PIL image, original size)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Load original image
    original = Image.open(path).convert("RGB")
    original_size = original.size  # (W, H)
    
    # Optionally resize if too large
    if max_size is not None:
        w, h = original.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            original = original.resize((new_w, new_h), Image.BICUBIC)
    
    # Convert to tensor
    tensor = TF.to_tensor(original).unsqueeze(0).to(device)
    
    return tensor, original, original_size


def save_image_full_res(
    tensor: torch.Tensor,
    path: Union[str, Path],
    original_size: Optional[Tuple[int, int]] = None,
    quality: int = 95
):
    """
    Save a tensor at full resolution, optionally resizing to original size.
    
    Args:
        tensor: Image tensor (1, 3, H, W) or (3, H, W)
        path: Output path
        original_size: Optional (W, H) to resize back to
        quality: JPEG quality
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.clamp(0, 1)
    pil_image = TF.to_pil_image(tensor.cpu())
    
    # Resize to original size if specified
    if original_size is not None:
        pil_image = pil_image.resize(original_size, Image.BICUBIC)
    
    # Save with appropriate format
    suffix = path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        pil_image.save(path, "JPEG", quality=quality)
    elif suffix == ".png":
        pil_image.save(path, "PNG")
    else:
        pil_image.save(path)
