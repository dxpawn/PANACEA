"""
VAE-based Adversarial Attack for Panacea

Uses Variational Autoencoder latent space perturbations for more natural-looking
adversarial examples. This approach modifies the latent representation rather
than raw pixels, producing perturbations that follow the image's natural structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from tqdm import tqdm


class SimpleVAE(nn.Module):
    """
    Simple Convolutional VAE for image perturbation.
    
    Architecture is lightweight (~2M params) to enable fast inference.
    The VAE learns to reconstruct images through a bottleneck, and we
    perturb the latent space to create adversarial examples.
    """
    
    def __init__(self, latent_dim: int = 256, image_size: int = 224):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 112 -> 56
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 56 -> 28
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 28 -> 14
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 14 -> 7
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 7 * 7)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 7 -> 14
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 14 -> 28
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 28 -> 56
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 56 -> 112
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 112 -> 224
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 7, 7)
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode -> sample -> decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEAttack:
    """
    VAE-based adversarial attack.
    
    Instead of perturbing pixels directly, we:
    1. Encode the image to latent space
    2. Perturb the latent vector
    3. Decode back to image space
    
    This produces perturbations that follow the natural image manifold,
    resulting in more natural-looking adversarial examples.
    """
    
    def __init__(
        self,
        clip_model,
        vae: Optional[SimpleVAE] = None,
        latent_epsilon: float = 2.0,
        step_size: float = 0.1,
        iterations: int = 100,
        pixel_weight: float = 0.3,
        verbose: bool = True
    ):
        """
        Initialize VAE attack.
        
        Args:
            clip_model: CLIP model wrapper for computing attack loss
            vae: Pre-trained VAE (if None, creates untrained VAE - less effective)
            latent_epsilon: Maximum perturbation in latent space (L2 norm)
            step_size: Step size for latent optimization
            iterations: Number of optimization iterations
            pixel_weight: Weight for pixel-space reconstruction constraint
            verbose: Whether to show progress bar
        """
        self.clip = clip_model
        self.device = clip_model.device
        self.latent_epsilon = latent_epsilon
        self.step_size = step_size
        self.iterations = iterations
        self.pixel_weight = pixel_weight
        self.verbose = verbose
        
        # Initialize or use provided VAE
        if vae is None:
            if verbose:
                print("   Creating lightweight VAE (untrained - for demo only)")
                print("   For better results, train VAE on image dataset first")
            self.vae = SimpleVAE().to(self.device)
            self.vae.eval()
            self._vae_trained = False
        else:
            self.vae = vae.to(self.device)
            self.vae.eval()
            self._vae_trained = True
    
    def targeted_attack(
        self,
        image: torch.Tensor,
        target_label: str,
        callback: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Targeted VAE attack: perturb latent space toward target class.
        
        Args:
            image: Input image (1, 3, 224, 224) in [0, 1]
            target_label: Target class to mimic
            callback: Optional progress callback
            
        Returns:
            Tuple of (perturbed image, metrics dict)
        """
        image = image.to(self.device)
        
        # Get target text features
        target_features = self.clip.get_text_features(target_label)
        
        # Encode image to latent space
        with torch.no_grad():
            mu, logvar = self.vae.encode(image)
            z_orig = mu.clone()  # Use mean for deterministic encoding
        
        # Initialize perturbation in latent space
        delta_z = torch.zeros_like(z_orig, requires_grad=True)
        
        # Get original similarity
        with torch.no_grad():
            orig_features = self.clip.get_image_features(image)
            orig_similarity = (orig_features @ target_features.T).item()
        
        iterator = range(self.iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc=f"VAE attack -> '{target_label}'")
        
        best_delta_z = delta_z.detach().clone()
        best_similarity = orig_similarity
        
        for i in iterator:
            delta_z.requires_grad_(True)
            
            # Decode perturbed latent
            z_perturbed = z_orig + delta_z
            reconstructed = self.vae.decode(z_perturbed)
            
            # Compute CLIP similarity to target
            img_features = self.clip.get_image_features_grad(reconstructed)
            similarity = img_features @ target_features.T
            
            # Pixel-space constraint (stay close to original)
            pixel_loss = F.mse_loss(reconstructed, image)
            
            # Combined loss: maximize similarity, minimize pixel change
            loss = -similarity.mean() + self.pixel_weight * pixel_loss
            
            loss.backward()
            
            with torch.no_grad():
                # Gradient descent in latent space
                delta_z = delta_z - self.step_size * delta_z.grad
                
                # Project onto L2 ball in latent space
                norm = delta_z.norm()
                if norm > self.latent_epsilon:
                    delta_z = delta_z * self.latent_epsilon / norm
            
            delta_z = delta_z.detach()
            
            current_sim = similarity.item()
            if current_sim > best_similarity:
                best_similarity = current_sim
                best_delta_z = delta_z.clone()
            
            if self.verbose:
                iterator.set_postfix(sim=f"{current_sim:.4f}", pix=f"{pixel_loss.item():.4f}")
        
        # Generate final image
        with torch.no_grad():
            final_z = z_orig + best_delta_z
            final_image = self.vae.decode(final_z)
            final_image = torch.clamp(final_image, 0, 1)
            
            final_features = self.clip.get_image_features(final_image)
            final_similarity = (final_features @ target_features.T).item()
        
        metrics = {
            "mode": "vae_targeted",
            "target_label": target_label,
            "original_similarity": orig_similarity,
            "final_similarity": final_similarity,
            "similarity_gain": final_similarity - orig_similarity,
            "latent_perturbation_norm": best_delta_z.norm().item(),
            "latent_epsilon": self.latent_epsilon,
            "iterations": self.iterations,
            "vae_trained": self._vae_trained,
        }
        
        return final_image, metrics
    
    def untargeted_attack(
        self,
        image: torch.Tensor,
        true_label: str,
        callback: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Untargeted VAE attack: perturb latent space away from true class.
        
        Args:
            image: Input image (1, 3, 224, 224) in [0, 1]
            true_label: True class to evade
            callback: Optional progress callback
            
        Returns:
            Tuple of (perturbed image, metrics dict)
        """
        image = image.to(self.device)
        
        # Get true label text features
        true_features = self.clip.get_text_features(true_label)
        
        # Encode image to latent space
        with torch.no_grad():
            mu, logvar = self.vae.encode(image)
            z_orig = mu.clone()
        
        # Initialize perturbation
        delta_z = torch.zeros_like(z_orig, requires_grad=True)
        
        # Get original similarity
        with torch.no_grad():
            orig_features = self.clip.get_image_features(image)
            orig_similarity = (orig_features @ true_features.T).item()
        
        iterator = range(self.iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc=f"VAE attack <- '{true_label}'")
        
        best_delta_z = delta_z.detach().clone()
        best_similarity = orig_similarity
        
        for i in iterator:
            delta_z.requires_grad_(True)
            
            # Decode perturbed latent
            z_perturbed = z_orig + delta_z
            reconstructed = self.vae.decode(z_perturbed)
            
            # Compute CLIP similarity to true label (minimize this)
            img_features = self.clip.get_image_features_grad(reconstructed)
            similarity = img_features @ true_features.T
            
            # Pixel-space constraint
            pixel_loss = F.mse_loss(reconstructed, image)
            
            # Combined loss: minimize similarity to true class
            loss = similarity.mean() + self.pixel_weight * pixel_loss
            
            loss.backward()
            
            with torch.no_grad():
                delta_z = delta_z - self.step_size * delta_z.grad
                norm = delta_z.norm()
                if norm > self.latent_epsilon:
                    delta_z = delta_z * self.latent_epsilon / norm
            
            delta_z = delta_z.detach()
            
            current_sim = similarity.item()
            if current_sim < best_similarity:
                best_similarity = current_sim
                best_delta_z = delta_z.clone()
            
            if self.verbose:
                iterator.set_postfix(sim=f"{current_sim:.4f}", pix=f"{pixel_loss.item():.4f}")
        
        # Generate final image
        with torch.no_grad():
            final_z = z_orig + best_delta_z
            final_image = self.vae.decode(final_z)
            final_image = torch.clamp(final_image, 0, 1)
            
            final_features = self.clip.get_image_features(final_image)
            final_similarity = (final_features @ true_features.T).item()
        
        metrics = {
            "mode": "vae_untargeted",
            "true_label": true_label,
            "original_similarity": orig_similarity,
            "final_similarity": final_similarity,
            "similarity_reduction": orig_similarity - final_similarity,
            "latent_perturbation_norm": best_delta_z.norm().item(),
            "latent_epsilon": self.latent_epsilon,
            "iterations": self.iterations,
            "vae_trained": self._vae_trained,
        }
        
        return final_image, metrics


def train_vae(
    vae: SimpleVAE,
    dataloader,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda",
    verbose: bool = True
) -> SimpleVAE:
    """
    Train VAE on an image dataset for better reconstruction quality.
    
    Args:
        vae: VAE model to train
        dataloader: PyTorch DataLoader with images
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        verbose: Whether to show progress
        
    Returns:
        Trained VAE model
    """
    vae = vae.to(device)
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            recon, mu, logvar = vae(images)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(recon, images, reduction='sum')
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + 0.001 * kl_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader.dataset):.4f}")
    
    vae.eval()
    return vae
