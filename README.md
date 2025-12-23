note to self: add vae-based attack later
untargeted pgd > targeted pgd, in defense mode, so theres no such thing as "targeted defense"
write mathematical explanation in latex

<h1 align="center">🛡️ Panacea</h1>

<p align="center">
Adversarial Image Perturbation System<br>
Invisible to Humans • Hostile to Models • Transferable by Design<br>
<b>Engineered for maximum model damage at minimal visual cost</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Attack-PGD%20%7C%20CLIP--Driven-ff5555?style=for-the-badge">
  <img src="https://img.shields.io/badge/Defense-Image%20Cloaking-4dabf7?style=for-the-badge">
  <img src="https://img.shields.io/badge/Poisoning-Targeted%20Feature%20Pull-9b59b6?style=for-the-badge">
  <img src="https://img.shields.io/badge/Perceptual-LPIPS%20%7C%20Saliency-2ecc71?style=for-the-badge">
  <img src="https://img.shields.io/badge/Backbone-CLIP--Based-f1c40f?style=for-the-badge">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Threat%20Model-Modern%20Diffusion%20Pipelines-critical?style=flat-square">
  <img src="https://img.shields.io/badge/Visibility-~35dB%20PSNR-success?style=flat-square">
  <img src="https://img.shields.io/badge/Transferability-Model--Agnostic-informational?style=flat-square">
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=14&pause=1200&color=36BCF7&center=true&vCenter=true&width=900&lines=CLIP--Aware+Adversarial+Perturbations;Invisible+to+Humans+Not+to+Models;Offense+%7C+Defense+%7C+Hybrid+Attacks;Protect+Your+Pixels+Before+They+Get+Scraped">
</p>

Panacea is a tool for protecting images from AI models through imperceptible adversarial perturbations. Similar to [Nightshade](https://nightshade.cs.uchicago.edu/) and [Glaze](https://glaze.cs.uchicago.edu/), it modifies images in ways that are invisible to humans but disrupt AI understanding.

---

## ✨ Features

### Two Attack Modes

| Mode | Purpose | How It Works |
|------|---------|--------------|
| **Targeted (Offense)** | Data poisoning | Pulls image features toward a fake target class. AI "sees" features that don't exist. |
| **Untargeted (Defense)** | Image cloaking | Pushes image features away from true class. AI can't recognize the real content. |

### Visual Quality Enhancements (v1.1)
- **LPIPS Perceptual Loss**: Uses VGG-based similarity to keep perturbations invisible (~35dB PSNR)
- **Saliency Masking**: Reduces perturbations on edges and important features
- **Hybrid Attack**: Combined push-and-pull for maximum disruption
- **CLIP-based**: Targets the backbone of modern AI art generators (Stable Diffusion, DALL-E, Midjourney)

## 🚀 Installation

```bash
# Clone or download the repository
cd Panacea

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for faster processing)

## 📖 Usage

### Quick Demo

```bash
python main.py demo
```

### Targeted Attack (Offense)
Make AI think your dog photo is "abstract art":

```bash
python main.py attack -i dog.png -o poisoned.png -m targeted -t "abstract art"
```

### Untargeted Attack (Defense)
Cloak your portrait so AI can't recognize it:

```bash
python main.py attack -i portrait.png -o cloaked.png -m untargeted -l "human face portrait"
```

### Hybrid Attack
Push away from "dog" and pull toward "cat":

```bash
python main.py attack -i dog.png -o hybrid.png -m hybrid -l "dog" -t "cat"
```

### Adjust Visual Quality

```bash
# Higher perceptual weight = more invisible, weaker attack (default: 0.3)
python main.py attack -i img.png -o out.png -m targeted -t "cat" -p 0.5

# Disable perceptual loss for faster processing
python main.py attack -i img.png -o out.png -m targeted -t "cat" --no-perceptual
```

### Analyze & Compare

```bash
# Check how CLIP perceives an image
python main.py analyze -i image.png -l "cat" -l "dog" -l "abstract art"

# Measure perturbation visibility
python main.py compare -o original.png -p perturbed.png
```

## ⚙️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epsilon`, `-e` | 0.05 | Max perturbation magnitude (L∞ bound). Higher = more effective but more visible. |
| `--iterations`, `-n` | 100 | Number of PGD optimization steps. More iterations = better attack. |
| `--step-size`, `-s` | 0.01 | Step size per iteration. |
| `--perceptual-weight`, `-p` | 0.3 | Weight for LPIPS loss (0-1). Higher = more invisible, weaker attack. |
| `--no-perceptual` | - | Disable LPIPS perceptual loss for faster processing. |
| `--no-saliency` | - | Disable saliency-based masking. |
| `--device`, `-d` | auto | `cuda` or `cpu`. Auto-detects GPU. |

## 🔬 How It Works

Panacea uses **Projected Gradient Descent (PGD)** with **LPIPS perceptual constraints**:

```
For each iteration:
    1. Compute CLIP embedding similarity
    2. Compute LPIPS perceptual loss
    3. Combine losses with perceptual weight
    4. Calculate gradients w.r.t. input pixels
    5. Apply saliency-weighted gradient update
    6. Project perturbation onto ε-ball (L∞ constraint)
    7. Clamp to valid pixel range [0, 1]
```

### Why CLIP?

CLIP is the backbone of most modern AI image generators:
- **Stable Diffusion** uses CLIP for text-image alignment
- **DALL-E** uses CLIP for image ranking
- **Midjourney** uses CLIP-like models

Perturbations effective against CLIP transfer well to these downstream models.

## 📊 Quality Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher = less visible perturbation
  - \>40 dB: Virtually invisible
  - 30-40 dB: Imperceptible to most viewers ← **Panacea v1.1 achieves ~35dB**
  - 20-30 dB: Subtle differences may be visible
  
- **LPIPS**: Lower = more perceptually similar (less visible)

- **L∞ norm**: Maximum pixel change. Bounded by epsilon parameter.

## 📁 Project Structure

```
Panacea/
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── README.md           # This file
└── panacea/
    ├── __init__.py     # Package initialization
    ├── models.py       # CLIP model wrapper
    ├── attacks.py      # PGD attack with perceptual loss
    ├── perceptual.py   # LPIPS and saliency masking
    ├── utils.py        # Image I/O and metrics
    └── cli.py          # Command line interface
```

## ⚠️ Ethical Considerations

This tool is designed for **legitimate defensive purposes**:

✅ **Appropriate Uses**:
- Protecting your own artwork from unauthorized AI training
- Research into adversarial robustness
- Understanding AI model vulnerabilities

❌ **Inappropriate Uses**:
- Applying to images you don't own
- Maliciously poisoning public datasets (I will find you and I will beat the shit out of you, personally)
- Bypassing content moderation systems

## 🔗 References

- [Nightshade](https://nightshade.cs.uchicago.edu/) - Data poisoning tool from UChicago
- [Glaze](https://glaze.cs.uchicago.edu/) - Style mimicry protection
- [CLIP](https://openai.com/research/clip) - OpenAI's vision-language model
- [LPIPS](https://richzhang.github.io/PerceptualSimilarity/) - Learned Perceptual Image Patch Similarity
- [PGD Attack](https://arxiv.org/abs/1706.06083) - Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"


## 📄 License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

You are free to:
- Use, study, and modify the source code
- Redistribute modified versions under the same license

Under the following conditions:
- **Derivative works must remain open-source under GPLv3**
- **Attribution is required**
- **No warranty is provided**

### Ethical Use Clause (repeated again because this is important)
This software is intended for **defensive, research, and self-protection purposes only**.

You **must not**:
- Use Panacea to poison datasets you do not own or control
- Deploy it at scale against public or community datasets
- Weaponize it for harassment, sabotage, or model vandalism

If you do, that’s on you - legally, ethically, and karmically.  
The author disclaims responsibility for misuse.

Use responsibly.

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/dxpawn">
        <img src="https://github.com/dxpawn.png" width="96" style="border-radius: 50%"><br />
        <sub><b>Nguyen Dan Vu</b></sub><br />
      </a>
      <sub>
        Lone Wolf<br/>
        <i>Professional AI Art Hater</i>
      </sub>
    </td>
  </tr>
</table>
