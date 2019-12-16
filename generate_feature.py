import torch
import torch.nn as nn
import argparse
import numpy as np
from nn import ResNet18
from torch.utils.data import DataLoader
from progressbar import ProgressBar
from torchvision import datasets, transforms
from tools import seed_everything

class ExtractFeature(nn.Module):
    def __init__(self, pretrained):
        super(ExtractFeature ,self).__init__()
        self.pretrained = pretrained
        self._reset_model()

    def _reset_model(self):
        model = ResNet18()
        model.load_state_dict(torch.load(model_path))
        self._features = nn.Sequential(
            model.conv1,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )
    def forward(self,inputs):
        out = self._features(inputs)
        out = torch.flatten(out,1)
        return out

def generate_feature(data_loader):
    extract_feature.eval()
    out_target = []
    out_output =[]
    pbar = ProgressBar(n_total=len(data_loader), desc='GenerateFeature')
    for batch_idx,(data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output = extract_feature(data)
        output_np = output.data.cpu().numpy()
        target_np = target.data.cpu().numpy()

        out_output.append(output_np)
        out_target.append(target_np[:, np.newaxis])
        pbar(step=batch_idx)
    output_array = np.concatenate(out_output, axis=0)
    target_array = np.concatenate(out_target, axis=0)
    np.save(f'./feature/{arch}_feature.npy', output_array, allow_pickle=False)
    np.save(f'./feature/{arch}_target.npy', target_array, allow_pickle=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10')
    parser.add_argument("--model", type=str, default='ResNet18')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epoch',type=int,default=30)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument("--task", type=str, default='image')
    parser.add_argument("--do_lsr", action='store_true',help="Whether to do label smoothing.")
    args = parser.parse_args()
    seed_everything(args.seed)

    if args.do_lsr:
        arch = args.model+'_label_smoothing'
    else:
        arch = args.model

    model_path = f"./checkpoints/{arch}.bin"
    extract_feature = ExtractFeature(model_path)
    device = torch.device("cuda:0")
    extract_feature.to(device)

    data = {
        'valid': datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))]
            )
        )
    }

    loaders = {
        'valid': DataLoader(data['valid'], batch_size=128,
                            num_workers=10, pin_memory=True,
                            drop_last=False)
    }
    generate_feature(loaders['valid'])
