"""
Adversarial Attack Algorithms for Panacea

Implements targeted and untargeted attacks using Projected Gradient Descent (PGD)
with perceptual loss constraints for improved visual quality.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Tuple, Callable
from .models import CLIPWrapper
from .perceptual import PerceptualLoss, SaliencyMask


class PanaceaAttack:
    """
    Adversarial perturbation attack using CLIP embeddings with perceptual constraints.
    
    Supports two modes:
    1. Targeted Attack (Offense): Pull image features toward a target class
    2. Untargeted Attack (Defense): Push image features away from true class
    
    Enhanced with LPIPS perceptual loss for improved visual quality.
    """
    
    def __init__(
        self,
        clip_model: CLIPWrapper,
        epsilon: float = 0.05,
        step_size: float = 0.01,
        iterations: int = 100,
        perceptual_weight: float = 0.5,
        use_perceptual: bool = True,
        use_saliency: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the attack.
        
        Args:
            clip_model: CLIP model wrapper
            epsilon: Maximum perturbation magnitude (L-infinity bound)
            step_size: Step size for each PGD iteration
            iterations: Number of PGD iterations
            perceptual_weight: Weight for perceptual loss (0-1). Higher = more invisible but weaker attack.
            use_perceptual: Whether to use LPIPS perceptual loss
            use_saliency: Whether to use saliency-based masking
            verbose: Whether to show progress bar
        """
        self.clip = clip_model
        self.epsilon = epsilon
        self.step_size = step_size
        self.iterations = iterations
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual
        self.use_saliency = use_saliency
        self.verbose = verbose
        self.device = clip_model.device
        
        # Initialize perceptual loss if enabled
        self.perceptual_loss = None
        if use_perceptual:
            if verbose:
                print("   Loading LPIPS perceptual model...")
            self.perceptual_loss = PerceptualLoss(device=self.device)
        
        # Initialize saliency mask if enabled
        self.saliency_mask = None
        if use_saliency:
            self.saliency_mask = SaliencyMask(device=self.device)
    
    def _compute_saliency_weight(self, image: torch.Tensor) -> torch.Tensor:
        """Compute saliency-based weight mask for perturbation."""
        if self.saliency_mask is None:
            return torch.ones_like(image)
        
        # Get edge mask (high values = safe to perturb)
        mask = self.saliency_mask.compute_edge_mask(image)
        
        # Apply soft weighting: allow some perturbation everywhere, but more in safe areas
        weight = 0.3 + 0.7 * mask
        
        return weight
    
    def targeted_attack(
        self,
        image: torch.Tensor,
        target_label: str,
        callback: Optional[Callable[[int, float], None]] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Targeted attack: Minimize distance to target class.
        
        This makes the AI "see" features of the target class that humans don't see.
        For example, making a picture of a dog look like a "cat" to AI.
        
        Args:
            image: Input image tensor of shape (1, 3, H, W) in range [0, 1]
            target_label: Target class label (e.g., "cat", "abstract art")
            callback: Optional callback function(iteration, loss)
            
        Returns:
            Tuple of (perturbed image, metrics dict)
        """
        image = image.to(self.device)
        
        # Get target text features
        target_features = self.clip.get_text_features(target_label)
        
        # Compute saliency weight
        saliency_weight = self._compute_saliency_weight(image)
        
        # Initialize perturbation
        delta = torch.zeros_like(image, requires_grad=True, device=self.device)
        
        # Get original features for comparison
        with torch.no_grad():
            orig_features = self.clip.get_image_features(image)
            orig_similarity = (orig_features @ target_features.T).item()
        
        iterator = range(self.iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc=f"Targeted attack -> '{target_label}'")
        
        best_delta = delta.detach().clone()
        best_score = float('-inf')
        
        for i in iterator:
            delta.requires_grad_(True)
            
            # Compute perturbed image
            perturbed = torch.clamp(image + delta, 0, 1)
            
            # Get image features (with gradients)
            img_features = self.clip.get_image_features_grad(perturbed)
            
            # Compute cosine similarity to target (we want to maximize this)
            similarity = img_features @ target_features.T
            
            # CLIP loss: negative similarity (minimize to maximize similarity)
            clip_loss = -similarity.mean()
            
            # Perceptual loss: keep perturbation invisible
            if self.perceptual_loss is not None:
                perc_loss = self.perceptual_loss(image, perturbed)
                # Combined loss: balance attack effectiveness and invisibility
                total_loss = (1 - self.perceptual_weight) * clip_loss + self.perceptual_weight * perc_loss
            else:
                total_loss = clip_loss
                perc_loss = torch.tensor(0.0)
            
            # Backward pass
            total_loss.backward()
            
            # PGD step with saliency weighting
            with torch.no_grad():
                # Gradient direction
                grad_sign = delta.grad.sign()
                
                # Apply saliency weighting to gradient
                weighted_grad = grad_sign * saliency_weight
                
                # Update perturbation
                delta = delta - self.step_size * weighted_grad
                
                # Project onto epsilon ball (L-infinity constraint)
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                
                # Ensure perturbed image is valid
                delta = torch.clamp(image + delta, 0, 1) - image
            
            delta = delta.detach()
            
            current_sim = -clip_loss.item()
            
            # Score: maximize similarity while minimizing perceptual loss
            if self.perceptual_loss is not None:
                score = current_sim - self.perceptual_weight * perc_loss.item()
            else:
                score = current_sim
            
            if score > best_score:
                best_score = score
                best_delta = delta.clone()
            
            if callback:
                callback(i, current_sim)
            
            if self.verbose:
                if self.perceptual_loss is not None:
                    iterator.set_postfix(sim=f"{current_sim:.4f}", perc=f"{perc_loss.item():.4f}")
                else:
                    iterator.set_postfix(similarity=f"{current_sim:.4f}")
        
        # Compute final perturbed image
        final_image = torch.clamp(image + best_delta, 0, 1)
        
        # Compute metrics
        with torch.no_grad():
            final_features = self.clip.get_image_features(final_image)
            final_similarity = (final_features @ target_features.T).item()
            final_perc = self.perceptual_loss(image, final_image).item() if self.perceptual_loss else 0.0
        
        metrics = {
            "mode": "targeted",
            "target_label": target_label,
            "original_similarity": orig_similarity,
            "final_similarity": final_similarity,
            "similarity_gain": final_similarity - orig_similarity,
            "perceptual_loss": final_perc,
            "epsilon": self.epsilon,
            "iterations": self.iterations,
            "perceptual_weight": self.perceptual_weight,
        }
        
        return final_image, metrics
    
    def untargeted_attack(
        self,
        image: torch.Tensor,
        true_label: str,
        callback: Optional[Callable[[int, float], None]] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Untargeted attack: Maximize distance from true class.
        
        This "cloaks" the image by pushing its features away from its true cluster.
        AI will have difficulty recognizing the image's true content.
        
        Args:
            image: Input image tensor of shape (1, 3, H, W) in range [0, 1]
            true_label: True class label of the image
            callback: Optional callback function(iteration, loss)
            
        Returns:
            Tuple of (perturbed image, metrics dict)
        """
        image = image.to(self.device)
        
        # Get true label text features
        true_features = self.clip.get_text_features(true_label)
        
        # Compute saliency weight
        saliency_weight = self._compute_saliency_weight(image)
        
        # Initialize perturbation
        delta = torch.zeros_like(image, requires_grad=True, device=self.device)
        
        # Get original features for comparison
        with torch.no_grad():
            orig_features = self.clip.get_image_features(image)
            orig_similarity = (orig_features @ true_features.T).item()
        
        iterator = range(self.iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc=f"Untargeted attack <- '{true_label}'")
        
        best_delta = delta.detach().clone()
        best_score = float('-inf')
        
        for i in iterator:
            delta.requires_grad_(True)
            
            # Compute perturbed image
            perturbed = torch.clamp(image + delta, 0, 1)
            
            # Get image features (with gradients)
            img_features = self.clip.get_image_features_grad(perturbed)
            
            # Compute cosine similarity to true label (we want to minimize this)
            similarity = img_features @ true_features.T
            
            # CLIP loss: positive similarity (minimize to push away)
            clip_loss = similarity.mean()
            
            # Perceptual loss: keep perturbation invisible
            if self.perceptual_loss is not None:
                perc_loss = self.perceptual_loss(image, perturbed)
                # For untargeted: we want to minimize similarity AND perceptual loss
                # So combined loss = clip_loss + perc_loss (both should be minimized)
                total_loss = (1 - self.perceptual_weight) * clip_loss + self.perceptual_weight * perc_loss
            else:
                total_loss = clip_loss
                perc_loss = torch.tensor(0.0)
            
            # Backward pass
            total_loss.backward()
            
            # PGD step with saliency weighting
            with torch.no_grad():
                # Gradient direction
                grad_sign = delta.grad.sign()
                
                # Apply saliency weighting to gradient
                weighted_grad = grad_sign * saliency_weight
                
                # Update perturbation
                delta = delta - self.step_size * weighted_grad
                
                # Project onto epsilon ball (L-infinity constraint)
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                
                # Ensure perturbed image is valid
                delta = torch.clamp(image + delta, 0, 1) - image
            
            delta = delta.detach()
            
            current_sim = clip_loss.item()
            
            # Score: minimize similarity AND perceptual loss (invert for max tracking)
            if self.perceptual_loss is not None:
                score = -current_sim - self.perceptual_weight * perc_loss.item()
            else:
                score = -current_sim
            
            if score > best_score:
                best_score = score
                best_delta = delta.clone()
            
            if callback:
                callback(i, current_sim)
            
            if self.verbose:
                if self.perceptual_loss is not None:
                    iterator.set_postfix(sim=f"{current_sim:.4f}", perc=f"{perc_loss.item():.4f}")
                else:
                    iterator.set_postfix(similarity=f"{current_sim:.4f}")
        
        # Compute final perturbed image
        final_image = torch.clamp(image + best_delta, 0, 1)
        
        # Compute metrics
        with torch.no_grad():
            final_features = self.clip.get_image_features(final_image)
            final_similarity = (final_features @ true_features.T).item()
            final_perc = self.perceptual_loss(image, final_image).item() if self.perceptual_loss else 0.0
        
        metrics = {
            "mode": "untargeted",
            "true_label": true_label,
            "original_similarity": orig_similarity,
            "final_similarity": final_similarity,
            "similarity_reduction": orig_similarity - final_similarity,
            "perceptual_loss": final_perc,
            "epsilon": self.epsilon,
            "iterations": self.iterations,
            "perceptual_weight": self.perceptual_weight,
        }
        
        return final_image, metrics
    
    def hybrid_attack(
        self,
        image: torch.Tensor,
        true_label: str,
        target_label: str,
        alpha: float = 0.5,
        callback: Optional[Callable[[int, float, float], None]] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Hybrid attack: Push away from true class AND pull toward target class.
        
        Combines both objectives with a weighting factor alpha.
        
        Args:
            image: Input image tensor of shape (1, 3, H, W) in range [0, 1]
            true_label: True class label of the image
            target_label: Target class label to mimic
            alpha: Weight for target loss (1-alpha for push-away loss)
            callback: Optional callback function(iteration, target_sim, true_sim)
            
        Returns:
            Tuple of (perturbed image, metrics dict)
        """
        image = image.to(self.device)
        
        # Get text features
        true_features = self.clip.get_text_features(true_label)
        target_features = self.clip.get_text_features(target_label)
        
        # Compute saliency weight
        saliency_weight = self._compute_saliency_weight(image)
        
        # Initialize perturbation
        delta = torch.zeros_like(image, requires_grad=True, device=self.device)
        
        # Get original features
        with torch.no_grad():
            orig_features = self.clip.get_image_features(image)
            orig_true_sim = (orig_features @ true_features.T).item()
            orig_target_sim = (orig_features @ target_features.T).item()
        
        iterator = range(self.iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc=f"Hybrid: '{true_label}' -> '{target_label}'")
        
        best_delta = delta.detach().clone()
        best_score = float('-inf')
        
        for i in iterator:
            delta.requires_grad_(True)
            
            # Compute perturbed image
            perturbed = torch.clamp(image + delta, 0, 1)
            
            # Get image features (with gradients)
            img_features = self.clip.get_image_features_grad(perturbed)
            
            # Compute similarities
            true_sim = img_features @ true_features.T
            target_sim = img_features @ target_features.T
            
            # Combined CLIP loss: maximize target similarity, minimize true similarity
            clip_loss = -alpha * target_sim.mean() + (1 - alpha) * true_sim.mean()
            
            # Perceptual loss
            if self.perceptual_loss is not None:
                perc_loss = self.perceptual_loss(image, perturbed)
                total_loss = (1 - self.perceptual_weight) * clip_loss + self.perceptual_weight * perc_loss
            else:
                total_loss = clip_loss
                perc_loss = torch.tensor(0.0)
            
            # Backward pass
            total_loss.backward()
            
            # PGD step with saliency weighting
            with torch.no_grad():
                grad_sign = delta.grad.sign()
                weighted_grad = grad_sign * saliency_weight
                delta = delta - self.step_size * weighted_grad
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                delta = torch.clamp(image + delta, 0, 1) - image
            
            delta = delta.detach()
            
            current_target = target_sim.item()
            current_true = true_sim.item()
            
            # Score includes perceptual loss penalty
            if self.perceptual_loss is not None:
                score = alpha * current_target - (1 - alpha) * current_true - self.perceptual_weight * perc_loss.item()
            else:
                score = alpha * current_target - (1 - alpha) * current_true
            
            if score > best_score:
                best_score = score
                best_delta = delta.clone()
            
            if callback:
                callback(i, current_target, current_true)
            
            if self.verbose:
                if self.perceptual_loss is not None:
                    iterator.set_postfix(tgt=f"{current_target:.3f}", true=f"{current_true:.3f}", perc=f"{perc_loss.item():.3f}")
                else:
                    iterator.set_postfix(target=f"{current_target:.3f}", true=f"{current_true:.3f}")
        
        # Compute final perturbed image
        final_image = torch.clamp(image + best_delta, 0, 1)
        
        # Compute metrics
        with torch.no_grad():
            final_features = self.clip.get_image_features(final_image)
            final_true_sim = (final_features @ true_features.T).item()
            final_target_sim = (final_features @ target_features.T).item()
            final_perc = self.perceptual_loss(image, final_image).item() if self.perceptual_loss else 0.0
        
        metrics = {
            "mode": "hybrid",
            "true_label": true_label,
            "target_label": target_label,
            "alpha": alpha,
            "original_true_similarity": orig_true_sim,
            "original_target_similarity": orig_target_sim,
            "final_true_similarity": final_true_sim,
            "final_target_similarity": final_target_sim,
            "perceptual_loss": final_perc,
            "epsilon": self.epsilon,
            "iterations": self.iterations,
            "perceptual_weight": self.perceptual_weight,
        }
        
        return final_image, metrics
