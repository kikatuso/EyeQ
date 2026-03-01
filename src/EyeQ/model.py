import torch
import torch.nn as nn
from torchvision.transforms.functional import resize
import glob 
from efficientnet_pytorch import EfficientNet

## class meanings:
## 0: good quality
## 1: usable quality
## 2: bad quality


class EyeQ(nn.Module):
    def __init__(self,checkpoint_path='checkpoints/efficientnet/',pretrained = True, lightweight=False,mode='eval',verbose=False,return_probs=False,resize=520):
        super(EyeQ, self).__init__()
        assert mode in ['train', 'eval'], "mode should be either 'train' or 'eval'"
        checkpoints = glob.glob(checkpoint_path + '**/*.pth', recursive=True)
        self.return_probs = return_probs
        self.resize = resize
        if lightweight:
            print('Using lightweight model, only loading one checkpoint') if verbose else None
            checkpoints = [checkpoints[0]] # only load one checkpoint 
        self.models = nn.ModuleList([Efficientnet_fl(pretrained) for _ in checkpoints])
        for i, checkpoint in enumerate(checkpoints):
            print(f'loading checkpoint {checkpoint}') if verbose else None
            weights = torch.load(checkpoint, map_location='cpu',weights_only=True)
            self.models[i].load_state_dict(weights, strict=True)
        if mode == 'eval':
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()
        print('Initialised EyeQ with {} parameters'.format(self._num_params())) if verbose else None
        
    def _num_params(self):
        return sum(p.numel() for model in self.models for p in model.parameters())
    
    def _preprocess(self,x):
        if self.resize is not None:
            x = resize(x, [self.resize, self.resize])
        mean_val = x[x > 0].mean()
        std_val = x[x > 0].std()
        x = (x - mean_val) / std_val
        return x
    
    def forward(self, x):
        x = self._preprocess(x)
        preds = [ nn.Softmax(dim=1)(model(x)) for model in self.models]
        average = torch.mean(torch.stack(preds), dim=0)
        print(average)
        if self.return_probs:
            return average
        else:
            return torch.argmax(average, dim=1)


def Efficientnet_fl(pretrained):
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Identity()
    net_fl = nn.Sequential(
            nn.Linear(1792, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 3)
            )
    model._fc = net_fl
    
    return model

