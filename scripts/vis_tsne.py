'''
Usage:

python scripts/vis_tsne.py \
    --task_group put-block-in-bowl-seen-colors \
    --model roboclip \
    --ckpt_path finetune/put-block-in-bowl-seen-colors-roboclip_RN50_FiLM/checkpoints/best-v1.ckpt
'''
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from nat_policies.datasets import RoboCLIPDataset
from nat_policies.utils.common import load_original_clip
from nat_policies.models.roboclip import RoboCLIP


DATA_DIR = os.path.join(os.environ['CLIPORT_ROOT'], 'data')
SAVE_DIR = os.path.join(os.environ['NAT_POLICIES_ROOT'], 'visuals')
CONFIG = {
    'dataset': {
        'cache': True,
        'images': True,
        'augment': {
            'theta_sigma': 60
        },
    },
}


def get_model(args):
    model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'clip':
        model = load_original_clip(variant='RN50', device=device)
    elif args.model == 'roboclip':
        model = RoboCLIP(clip_variant='RN50', device=device)
        model.inference_mode(ckpt_path=args.ckpt_path)

    model.eval()
    return model, device


def main(args):
    train_ds = RoboCLIPDataset(
        DATA_DIR, CONFIG, group=args.task_group, mode='train',
        n_demos=args.n_demos, augment=False
    )
    model, device = get_model(args)

    start_imgs = []
    goal_imgs = []
    n_samples = args.n_demos
    for i in range(n_samples):
        print(f'Sampling data: {i+1}/{n_samples}')
        sample = train_ds.__getitem__(i, choose_random=False)
        start_imgs.append(sample['start_img'])
        goal_imgs.append(sample['goal_img'])
    
    start_imgs = torch.stack(start_imgs, dim=0).to(device)
    goal_imgs = torch.stack(goal_imgs, dim=0).to(device)
    print(f'Computing embeddings')
    with torch.no_grad():
        start_embs = model.encode_image(start_imgs)
        goal_embs = model.encode_image(goal_imgs)

    print(f'Fitting tSNE')
    start_embs = start_embs.cpu().numpy()
    n_starts = len(start_embs)
    goal_embs = goal_embs.cpu().numpy()

    tsne_input = np.concatenate((start_embs, goal_embs), axis=0)
    tsne_output = TSNE(n_components=2, perplexity=n_samples, init='pca', learning_rate='auto').fit_transform(tsne_input)

    plt.scatter(tsne_output[:n_starts, 0], tsne_output[:n_starts, 1], c='red', label='start embeddings')
    plt.scatter(tsne_output[n_starts:, 0], tsne_output[n_starts:, 1], c='#e6a91d', label='goal embeddings')
    plt.legend(loc='upper left')
    plt.title(f'tSNE Embeddings {args.model}')
    plt.savefig(os.path.join(SAVE_DIR, f'tsne_embeddings_{args.model}.png'))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_group', type=str, required=True)
    parser.add_argument('--n_demos', type=int, default=30)
    parser.add_argument('--model', type=str, required=True, choices=['clip', 'roboclip'])
    parser.add_argument('--ckpt_path', type=str, default=None)
    return parser.parse_args()  


if __name__ == '__main__':
    main(get_args())