"""
Panacea - Adversarial Image Perturbation System

A tool for protecting images from AI models through adversarial perturbations.
Similar to Nightshade and Glaze.
"""

__version__ = "1.1.0"
__author__ = "Project Panacea"

from .attacks import PanaceaAttack
from .models import load_clip_model
from .utils import load_image, save_image
from .perceptual import PerceptualLoss

__all__ = ["PanaceaAttack", "load_clip_model", "load_image", "save_image", "PerceptualLoss"]
