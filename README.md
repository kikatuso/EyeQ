# EyeQ — Retinal Image Quality Grading

A clean, easy-to-use PyTorch wrapper for automated retinal fundus image quality grading. This package restructures the original [EyeQ](https://github.com/HzFu/EyeQ) project into a simple Python class, with checkpoints trained by [AutoMorph](https://github.com/rmaphoh/AutoMorph).

Images are automatically sorted into `good_quality/` and `bad_quality/` subdirectories within your input folder.

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/kikatuso/EyeQ.git
cd EyeQ
pip install -e .
```

### Requirements

- Python 3.7+
- PyTorch (with optional CUDA support)
- torchvision
- tqdm

---

## Quick Start

```python
import EyeQ

EyeQ.run_grading("/path/to/your/images")
```

After running, your directory will contain two new subfolders:

```
/path/to/your/images/
├── good_quality/
│   └── image1.png
├── bad_quality/
│   └── image2.png
└── ...
```

---

## Usage

### `run_grading(dir_path, ...)`

Scans a directory for retinal images, grades their quality using a pretrained EfficientNet model, and moves each image into either a `good_quality/` or `bad_quality/` subdirectory.

```python
EyeQ.run_grading(
    dir_path="/path/to/images",
    img_extension=".png",
    batch_size=16,
    verbose=False,
    resize=520,
    lightweight=False
)
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `dir_path` | `str` or `Path` | *(required)* | Path to the directory containing retinal images. |
| `img_extension` | `str` | `'.png'` | File extension to search for (e.g. `'.jpg'`, `'.tiff'`). |
| `batch_size` | `int` | `16` | Number of images processed per batch. Reduce if running out of GPU/CPU memory. |
| `verbose` | `bool` | `False` | If `True`, prints per-image grading results to stdout. |
| `resize` | `int` | `520` | Resolution images are resized to before inference (square). |
| `lightweight` | `bool` | `False` | If `True`, uses a lighter a single checkpoint variant for faster inference. |

### Notes

- Images already inside `good_quality/` or `bad_quality/` subdirectories are automatically skipped.
- GPU is used automatically when available via `torch.cuda.is_available()`.
- An image classified as "uncertain" (class 1) is still moved to `good_quality` unless the probability of bad quality exceeds 0.25.

---

## Example

```python
import EyeQ

# Basic usage
EyeQ.run_grading("/data/fundus_images")

# With custom options
EyeQ.run_grading(
    dir_path="/data/fundus_images",
    img_extension=".jpg",
    batch_size=8,
    verbose=True,
    resize=448,
    lightweight=True
)
```

---

## Model & Checkpoints

Pretrained checkpoints (EfficientNet-based) are included under `checkpoints/efficientnet/` and are loaded automatically. The checkpoints are compatible with those produced by the [AutoMorph](https://github.com/rmaphoh/AutoMorph) pipeline.

---

## Credits

This project is a refactored and repackaged version of:

- **EyeQ** by HzFu — [https://github.com/HzFu/EyeQ](https://github.com/HzFu/EyeQ)
- **AutoMorph** by rmaphoh — [https://github.com/rmaphoh/AutoMorph](https://github.com/rmaphoh/AutoMorph)

Please consider citing the original works if you use this in research.

---

## License

This project inherits the license of the original [EyeQ](https://github.com/HzFu/EyeQ) repository. Please refer to that repository for full license details.
