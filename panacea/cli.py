"""
Command Line Interface for Panacea

Provides CLI commands for running adversarial attacks.
"""

import click
import torch
from pathlib import Path
import json

from .models import load_clip_model
from .attacks import PanaceaAttack
from .utils import load_image, save_image, compute_psnr, compute_linf_norm


@click.group()
@click.version_option(version="1.2.0")
def cli():
    """Panacea - Adversarial Image Perturbation System
    
    Protect your images from AI models using imperceptible perturbations.
    """
    pass


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Path to input image")
@click.option("--output", "-o", "output_path", required=True, type=click.Path(),
              help="Path to save perturbed image")
@click.option("--mode", "-m", type=click.Choice(["targeted", "untargeted", "hybrid"]),
              default="untargeted", help="Attack mode")
@click.option("--target", "-t", default=None, help="Target label for targeted/hybrid attack")
@click.option("--label", "-l", default=None, help="True label for untargeted/hybrid attack")
@click.option("--epsilon", "-e", default=0.05, type=float,
              help="Maximum perturbation magnitude (0-1 range)")
@click.option("--iterations", "-n", default=100, type=int,
              help="Number of optimization iterations")
@click.option("--step-size", "-s", default=0.01, type=float,
              help="Step size for each iteration")
@click.option("--alpha", "-a", default=0.5, type=float,
              help="Weight for hybrid mode (0-1)")
@click.option("--device", "-d", default=None,
              help="Device to use (cuda/cpu, auto-detect if not specified)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress bar")
@click.option("--save-metrics", type=click.Path(), default=None,
              help="Path to save metrics JSON")
@click.option("--perceptual-weight", "-p", default=0.3, type=float,
              help="Weight for perceptual loss (0-1). Higher = more invisible but weaker attack.")
@click.option("--no-perceptual", is_flag=True, help="Disable LPIPS perceptual loss")
@click.option("--no-saliency", is_flag=True, help="Disable saliency-based masking")
def attack(input_path, output_path, mode, target, label, epsilon, iterations,
           step_size, alpha, device, quiet, save_metrics, perceptual_weight,
           no_perceptual, no_saliency):
    """Apply adversarial perturbation to an image.
    
    Examples:
    
    \b
    Targeted attack (make AI see "abstract art"):
        panacea attack -i photo.png -o protected.png -m targeted -t "abstract art"
    
    \b
    Untargeted attack (cloak a portrait):
        panacea attack -i portrait.png -o cloaked.png -m untargeted -l "human face"
    
    \b
    Hybrid attack (push from dog, pull to cat):
        panacea attack -i dog.png -o poison.png -m hybrid -l "dog" -t "cat"
    """
    # Validate arguments
    if mode == "targeted" and target is None:
        raise click.UsageError("Targeted mode requires --target/-t option")
    if mode == "untargeted" and label is None:
        raise click.UsageError("Untargeted mode requires --label/-l option")
    if mode == "hybrid" and (target is None or label is None):
        raise click.UsageError("Hybrid mode requires both --target/-t and --label/-l options")
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    click.echo(f" Device: {device}")
    click.echo(f" Loading image: {input_path}")
    
    # Load image
    image, _ = load_image(input_path)
    image = image.to(device)
    
    click.echo(" Loading CLIP model...")
    clip_model = load_clip_model(device=device)
    
    # Create attack
    attacker = PanaceaAttack(
        clip_model=clip_model,
        epsilon=epsilon,
        step_size=step_size,
        iterations=iterations,
        perceptual_weight=perceptual_weight,
        use_perceptual=not no_perceptual,
        use_saliency=not no_saliency,
        verbose=not quiet
    )
    
    # Run attack
    click.echo(f" Running {mode} attack...")
    
    if mode == "targeted":
        result, metrics = attacker.targeted_attack(image, target)
    elif mode == "untargeted":
        result, metrics = attacker.untargeted_attack(image, label)
    else:  # hybrid
        result, metrics = attacker.hybrid_attack(image, label, target, alpha=alpha)
    
    # Compute quality metrics
    psnr = compute_psnr(image.cpu(), result.cpu())
    linf = compute_linf_norm(image.cpu(), result.cpu())
    metrics["psnr_db"] = psnr
    metrics["linf_norm"] = linf
    
    # Save result
    save_image(result, output_path)
    click.echo(f" Saved perturbed image: {output_path}")
    
    # Print metrics
    click.echo("\n Metrics:")
    click.echo(f"   PSNR: {psnr:.2f} dB (higher = less visible)")
    if not no_perceptual:
        click.echo(f"   LPIPS: {metrics.get('perceptual_loss', 0):.4f} (lower = less visible)")
    click.echo(f"   L∞ norm: {linf:.4f} (epsilon bound: {epsilon})")
    
    if mode == "targeted":
        click.echo(f"   Similarity to '{target}': {metrics['original_similarity']:.4f} → {metrics['final_similarity']:.4f}")
    elif mode == "untargeted":
        click.echo(f"   Similarity to '{label}': {metrics['original_similarity']:.4f} → {metrics['final_similarity']:.4f}")
    else:
        click.echo(f"   Similarity to '{label}': {metrics['original_true_similarity']:.4f} → {metrics['final_true_similarity']:.4f}")
        click.echo(f"   Similarity to '{target}': {metrics['original_target_similarity']:.4f} → {metrics['final_target_similarity']:.4f}")
    
    # Save metrics if requested
    if save_metrics:
        with open(save_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        click.echo(f" Metrics saved: {save_metrics}")
    
    click.echo("\n Done!")


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Path to input image")
@click.option("--labels", "-l", multiple=True, required=True,
              help="Labels to compute similarity against (can specify multiple)")
@click.option("--device", "-d", default=None,
              help="Device to use (cuda/cpu)")
def analyze(input_path, labels, device):
    """Analyze image similarity to given labels using CLIP.
    
    Example:
        panacea analyze -i image.png -l "cat" -l "dog" -l "abstract art"
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    click.echo(f" Loading image: {input_path}")
    image, _ = load_image(input_path)
    image = image.to(device)
    
    click.echo(" Loading CLIP model...")
    clip_model = load_clip_model(device=device)
    
    # Get image features
    img_features = clip_model.get_image_features(image)
    
    # Compute similarities
    click.echo("\n Similarities:")
    results = []
    for label in labels:
        text_features = clip_model.get_text_features(label)
        sim = (img_features @ text_features.T).item()
        results.append((label, sim))
    
    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    
    for label, sim in results:
        bar = "█" * int(sim * 20) + "░" * (20 - int(sim * 20))
        click.echo(f"   [{bar}] {sim:.4f} - {label}")


@cli.command()
@click.option("--original", "-o", required=True, type=click.Path(exists=True),
              help="Path to original image")
@click.option("--perturbed", "-p", required=True, type=click.Path(exists=True),
              help="Path to perturbed image")
def compare(original, perturbed):
    """Compare original and perturbed images.
    
    Computes quality metrics to assess perturbation visibility.
    """
    click.echo(f" Loading images...")
    
    orig_tensor, _ = load_image(original)
    pert_tensor, _ = load_image(perturbed)
    
    psnr = compute_psnr(orig_tensor, pert_tensor)
    linf = compute_linf_norm(orig_tensor, pert_tensor)
    
    click.echo("\n Quality Metrics:")
    click.echo(f"   PSNR: {psnr:.2f} dB")
    click.echo(f"   L∞ norm: {linf:.4f}")
    
    # Interpretation
    click.echo("\n Interpretation:")
    if psnr > 40:
        click.echo("   ✅ Excellent quality - perturbation is virtually invisible")
    elif psnr > 30:
        click.echo("   ✅ Acceptable quality - perturbation is imperceptible to most viewers")
    elif psnr > 20:
        click.echo("   ⚠️ Subpar quality - subtle differences may be visible")
    else:
        click.echo("   ❌ Poor quality - perturbation may be noticeable")


@cli.command()
@click.option("--device", "-d", default=None, help="Device to use (cuda/cpu)")
@click.option("--no-perceptual", is_flag=True, help="Disable LPIPS for faster demo")
def demo(device, no_perceptual):
    """Run a demonstration with a test image.
    
    Creates a synthetic test image and runs both targeted and untargeted attacks.
    """
    import torch
    
    click.echo("=" * 60)
    click.echo(" Panacea - Adversarial Image Perturbation System")
    click.echo("=" * 60)
    click.echo()
    
    # Check for CUDA
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    click.echo(f" Using device: {device}")
    
    # Load CLIP model
    click.echo(" Loading CLIP model (ViT-B-32)...")
    clip_model = load_clip_model(device=device)
    click.echo("    Model loaded successfully!")
    
    # Create a simple test image (gradient)
    click.echo("\n Creating test image...")
    test_image = torch.zeros(1, 3, 224, 224, device=device)
    # Create a gradient pattern
    for i in range(224):
        test_image[0, 0, i, :] = i / 224  # Red gradient
        test_image[0, 1, :, i] = i / 224  # Green gradient
    test_image[0, 2, :, :] = 0.5  # Blue constant
    
    # Initialize attacker
    click.echo("\n  Initializing attack module...")
    attacker = PanaceaAttack(
        clip_model=clip_model,
        epsilon=0.05,
        step_size=0.01,
        iterations=50,  # Fewer iterations for demo
        perceptual_weight=0.3,
        use_perceptual=not no_perceptual,
        use_saliency=True,
        verbose=True
    )
    
    # Demo 1: Targeted Attack
    click.echo("\n" + "=" * 60)
    click.echo("Demo 1: Targeted Attack (Offense)")
    click.echo("Making the image look like 'a beautiful sunset' to AI")
    click.echo("=" * 60)
    
    result_targeted, metrics_targeted = attacker.targeted_attack(
        test_image,
        target_label="a beautiful sunset"
    )
    
    psnr_t = compute_psnr(test_image.cpu(), result_targeted.cpu())
    click.echo(f"\n Results:")
    click.echo(f"   Original similarity to 'sunset': {metrics_targeted['original_similarity']:.4f}")
    click.echo(f"   Final similarity to 'sunset':    {metrics_targeted['final_similarity']:.4f}")
    click.echo(f"   Similarity gain: {metrics_targeted['similarity_gain']:.4f}")
    click.echo(f"   PSNR: {psnr_t:.2f} dB")
    if not no_perceptual:
        click.echo(f"   LPIPS: {metrics_targeted['perceptual_loss']:.4f}")
    
    # Demo 2: Untargeted Attack
    click.echo("\n" + "=" * 60)
    click.echo("Demo 2: Untargeted Attack (Defense)")
    click.echo("Cloaking the image to hide from 'gradient pattern' detection")
    click.echo("=" * 60)
    
    result_untargeted, metrics_untargeted = attacker.untargeted_attack(
        test_image,
        true_label="colorful gradient pattern"
    )
    
    psnr_u = compute_psnr(test_image.cpu(), result_untargeted.cpu())
    click.echo(f"\n Results:")
    click.echo(f"   Original similarity to 'gradient': {metrics_untargeted['original_similarity']:.4f}")
    click.echo(f"   Final similarity to 'gradient':    {metrics_untargeted['final_similarity']:.4f}")
    click.echo(f"   Similarity reduction: {metrics_untargeted['similarity_reduction']:.4f}")
    click.echo(f"   PSNR: {psnr_u:.2f} dB")
    if not no_perceptual:
        click.echo(f"   LPIPS: {metrics_untargeted['perceptual_loss']:.4f}")
    
    click.echo("\n" + "=" * 60)
    click.echo(" Demo completed successfully!")
    click.echo("=" * 60)
    click.echo("\nTo process your own images, use:")
    click.echo("  python main.py attack -i input.png -o output.png -m targeted -t 'target label'")
    click.echo("  python main.py attack -i input.png -o output.png -m untargeted -l 'true label'")


@cli.command("vae-attack")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Path to input image")
@click.option("--output", "-o", "output_path", required=True, type=click.Path(),
              help="Path to save perturbed image")
@click.option("--mode", "-m", type=click.Choice(["targeted", "untargeted"]),
              default="untargeted", help="Attack mode")
@click.option("--target", "-t", default=None, help="Target label for targeted attack")
@click.option("--label", "-l", default=None, help="True label for untargeted attack")
@click.option("--latent-epsilon", default=2.0, type=float,
              help="Maximum perturbation in latent space (L2 norm)")
@click.option("--iterations", "-n", default=100, type=int,
              help="Number of optimization iterations")
@click.option("--device", "-d", default=None, help="Device to use (cuda/cpu)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress bar")
def vae_attack_cmd(input_path, output_path, mode, target, label, latent_epsilon,
                   iterations, device, quiet):
    """Apply VAE-based adversarial attack using latent space perturbation.
    
    VAE attacks perturb the latent representation rather than raw pixels,
    producing more natural-looking adversarial examples that follow the
    image manifold.
    
    NOTE: For best results, train the VAE on a large image dataset first.
    The default untrained VAE is for demonstration only.
    
    Examples:
    
    \b
    VAE targeted attack:
        panacea vae-attack -i photo.png -o protected.png -m targeted -t "abstract art"
    
    \b
    VAE untargeted attack:
        panacea vae-attack -i portrait.png -o cloaked.png -m untargeted -l "human face"
    """
    from .vae_attack import VAEAttack
    
    # Validate arguments
    if mode == "targeted" and target is None:
        raise click.UsageError("Targeted mode requires --target/-t option")
    if mode == "untargeted" and label is None:
        raise click.UsageError("Untargeted mode requires --label/-l option")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    click.echo(f" Device: {device}")
    click.echo(f" Loading image: {input_path}")
    
    image, _ = load_image(input_path)
    image = image.to(device)
    
    click.echo(" Loading CLIP model...")
    clip_model = load_clip_model(device=device)
    
    click.echo(" Initializing VAE attack...")
    attacker = VAEAttack(
        clip_model=clip_model,
        latent_epsilon=latent_epsilon,
        iterations=iterations,
        verbose=not quiet
    )
    
    click.echo(f"  Running VAE {mode} attack...")
    
    if mode == "targeted":
        result, metrics = attacker.targeted_attack(image, target)
    else:
        result, metrics = attacker.untargeted_attack(image, label)
    
    psnr = compute_psnr(image.cpu(), result.cpu())
    metrics["psnr_db"] = psnr
    
    save_image(result, output_path)
    click.echo(f" Saved: {output_path}")
    
    click.echo("\n Metrics:")
    click.echo(f"   PSNR: {psnr:.2f} dB")
    click.echo(f"   Latent perturbation: {metrics['latent_perturbation_norm']:.4f}")
    
    if mode == "targeted":
        click.echo(f"   Similarity to '{target}': {metrics['original_similarity']:.4f} → {metrics['final_similarity']:.4f}")
    else:
        click.echo(f"   Similarity to '{label}': {metrics['original_similarity']:.4f} → {metrics['final_similarity']:.4f}")
    
    click.echo("\n Done!")


@cli.command("attack-fullres")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Path to input image (any resolution)")
@click.option("--output", "-o", "output_path", required=True, type=click.Path(),
              help="Path to save perturbed image")
@click.option("--mode", "-m", type=click.Choice(["targeted", "untargeted"]),
              default="untargeted", help="Attack mode")
@click.option("--target", "-t", default=None, help="Target label for targeted attack")
@click.option("--label", "-l", default=None, help="True label for untargeted attack")
@click.option("--epsilon", "-e", default=0.05, type=float,
              help="Maximum perturbation magnitude")
@click.option("--iterations", "-n", default=100, type=int,
              help="Number of optimization iterations per tile")
@click.option("--max-size", default=None, type=int,
              help="Optional max dimension (preserves aspect ratio)")
@click.option("--overlap", default=32, type=int,
              help="Overlap between tiles for blending")
@click.option("--device", "-d", default=None, help="Device to use")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress bar")
def attack_fullres_cmd(input_path, output_path, mode, target, label, epsilon,
                       iterations, max_size, overlap, device, quiet):
    """Apply attack at full image resolution using tiled processing.
    
    Processes the image in overlapping 224x224 tiles, applies attacks,
    and blends the results back together. This preserves image resolution
    and quality while still applying effective attacks.
    
    Examples:
    
    \b
    Full-res targeted attack:
        panacea attack-fullres -i highres.jpg -o protected.jpg -m targeted -t "abstract"
    
    \b
    With max dimension limit:
        panacea attack-fullres -i huge.png -o out.png -m untargeted -l "portrait" --max-size 2048
    """
    from .full_resolution import FullResolutionProcessor, load_image_full_res, save_image_full_res
    
    # Validate arguments
    if mode == "targeted" and target is None:
        raise click.UsageError("Targeted mode requires --target/-t option")
    if mode == "untargeted" and label is None:
        raise click.UsageError("Untargeted mode requires --label/-l option")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    click.echo(f" Device: {device}")
    click.echo(f" Loading image at full resolution: {input_path}")
    
    image, original_pil, original_size = load_image_full_res(input_path, max_size=max_size, device=device)
    h, w = image.shape[2], image.shape[3]
    click.echo(f"   Image size: {w}x{h}")
    
    click.echo(" Loading CLIP model...")
    clip_model = load_clip_model(device=device)
    
    click.echo(" Initializing tile processor...")
    processor = FullResolutionProcessor(tile_size=224, overlap=overlap, device=device)
    tiles, positions = processor.extract_tiles(image)
    click.echo(f"   Split into {len(positions)} tiles")
    
    click.echo("  Initializing attack...")
    attacker = PanaceaAttack(
        clip_model=clip_model,
        epsilon=epsilon,
        iterations=iterations,
        use_perceptual=False,  # Faster for many tiles
        use_saliency=True,
        verbose=False
    )
    
    # Process each tile
    click.echo(f" Processing tiles ({mode} attack)...")
    perturbed_tiles = []
    
    iterator = range(len(positions))
    if not quiet:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Processing tiles")
    
    for i in iterator:
        tile = tiles[i:i+1]
        
        if mode == "targeted":
            result, _ = attacker.targeted_attack(tile, target)
        else:
            result, _ = attacker.untargeted_attack(tile, label)
        
        perturbed_tiles.append(result)
    
    perturbed_tiles = torch.cat(perturbed_tiles, dim=0)
    
    click.echo("🔗 Combining tiles...")
    final_image = processor.combine_tiles(perturbed_tiles, positions, (h, w))
    
    # Ensure perturbation is bounded
    delta = final_image - image
    delta = torch.clamp(delta, -epsilon, epsilon)
    final_image = torch.clamp(image + delta, 0, 1)
    
    psnr = compute_psnr(image.cpu(), final_image.cpu())
    
    save_image_full_res(final_image, output_path)
    click.echo(f" Saved: {output_path}")
    
    click.echo("\n Metrics:")
    click.echo(f"   Output resolution: {w}x{h}")
    click.echo(f"   PSNR: {psnr:.2f} dB")
    click.echo(f"   Tiles processed: {len(positions)}")
    
    click.echo("\n Done!")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
