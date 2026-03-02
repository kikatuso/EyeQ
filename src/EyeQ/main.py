from pathlib import Path
import torch
import shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from .model import EyeQ


def run_grading(dir_path, img_extension='.png', batch_size=16, verbose=False,resize=520,lightweight = False,min_resolution=None,filter_num_workers = 32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dir_path = Path(dir_path).resolve()
    good_quality_dir = dir_path / 'good_quality'
    bad_quality_dir = dir_path / 'bad_quality'
    good_quality_dir.mkdir(exist_ok=True)
    bad_quality_dir.mkdir(exist_ok=True)

    img_paths = [
        p for p in dir_path.rglob(f'*{img_extension}')
        if p.parent.name not in ('good_quality', 'bad_quality',"corrupted",'too_small')]

    print(f'Found {len(img_paths)} images in {dir_path}')
    img_paths,invalid = filter_images(img_paths,min_resolution = min_resolution,num_workers=filter_num_workers)
    print(f"Filtered {len(invalid)} corrupted images")

    move_invalid(invalid,dir_path)
   
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor()
    ])
 
    model = EyeQ(lightweight=lightweight,verbose=verbose).to(device)
    model.eval()

    dataset = SimpleDataset(img_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch, paths in tqdm(dataloader, desc='Grading images'):
            batch = batch.to(device)
            preds, probs = model(batch)

            for pred, prob, img_path in zip(preds, probs, paths):
                img_path = Path(img_path)
                cl = pred.item()
                bad_prob = prob[2].item()  # probability of bad quality
                if cl == 0 or (cl == 1 and bad_prob < 0.25):
                    print(f'{img_path}: good quality') if verbose else None
                    dest_path = good_quality_dir / img_path.name
                else:
                    print(f'{img_path}: bad quality') if verbose else None
                    dest_path = bad_quality_dir / img_path.name
                img_path.rename(dest_path)


def check_image(p, min_resolution):
    try:
        with Image.open(p) as img:
            w, h = img.size

        if min_resolution is not None and (w < min_resolution or h < min_resolution):
            return p, "small"

        return p, None

    except Exception:
        return p, "corrupted"

def filter_images(paths, min_resolution=None, num_workers=32):
    valid = []
    invalid = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for p, flag in tqdm(
            ex.map(lambda p: check_image(p, min_resolution), paths),
            total=len(paths),
            desc="Filtering images"
        ):
            if flag is None:
                valid.append(p)
            else:
                invalid.append((p, flag))
    return valid, invalid
    

def move_one(args):
    p, reason, corrupted_dir, small_dir = args
    target_dir = corrupted_dir if reason == "corrupted" else small_dir
    p.rename(target_dir / p.name)

def move_invalid(invalid, base_dir, num_workers=16):
    base_dir = Path(base_dir)
    corrupted_dir = base_dir / "corrupted"
    small_dir = base_dir / "too_small"
    corrupted_dir.mkdir(exist_ok=True)
    small_dir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(tqdm(
            ex.map(
                move_one,
                [(p, r, corrupted_dir, small_dir) for p, r in invalid]
            ),
            total=len(invalid),
            desc="Moving invalid images"
        ))


class SimpleDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, str(img_path)

