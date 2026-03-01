from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from .model import EyeQ


def run_grading(dir_path, img_extension='.png', batch_size=16, verbose=False,resize=520,lightweight = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dir_path = Path(dir_path).resolve()

    img_paths = [
        p for p in dir_path.rglob(f'*{img_extension}')
        if p.parent.name not in ('good_quality', 'bad_quality')
    ]

    good_quality_dir = dir_path / 'good_quality'
    bad_quality_dir = dir_path / 'bad_quality'
    good_quality_dir.mkdir(exist_ok=True)
    bad_quality_dir.mkdir(exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor()
    ])
 
    model = EyeQ(lightweight=lightweight,verbose=verbose).to(device)
    model.eval()
    if verbose:
        print(f'Found {len(img_paths)} images in {dir_path}')

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