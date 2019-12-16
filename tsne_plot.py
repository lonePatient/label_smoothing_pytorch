import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument("--model", type=str, default='ResNet18')
parser.add_argument("--task", type=str, default='image')
parser.add_argument("--do_lsr", action='store_true',help="Whether to do label smoothing.")
args = parser.parse_args()

if args.do_lsr:
    arch = args.model+'_label_smoothing'
else:
    arch = args.model
feature = np.load(f'./feature/{arch}_feature.npy').astype(np.float64)
target = np.load(f'./feature/{arch}_target.npy')
print('target shape: ', target.shape)
print('feature shape: ', feature.shape)

tsne = TSNE(n_components=2, init='pca', random_state=0)
output_2d = tsne.fit_transform(feature)
plt.rcParams['figure.figsize'] = 10, 10
plt.scatter(output_2d[:, 0], output_2d[:, 1], c= target[:,0])
plt.title(f"Validation {arch} tsne")
plt.savefig(f'./png/{arch}_feature_2d.png', bbox_inches='tight')
plt.show()
