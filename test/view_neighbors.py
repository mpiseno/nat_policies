import os
import random

from PIL import Image

import torch
import clip
import numpy as np
import matplotlib.pyplot as plt

from nat_policies.datasets import CLIPortDataset
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors


N_QUERY = 8
K = 5
GOAL_TYPE = 'combined'
OBJECTIVE = 'L2'
SPLIT = 'val'
EXTRA = ''
DATA_PATH = f'data/put-block-in-bowl-seen-colors-val'
MODEL_FP = 'logs/firm-puddle-23/ckpt_epoch=100.pt'

def get_full_dataset(data_path, preprocess):
    triple_filepaths = sorted(os.listdir(data_path))
    start_imgs, lang_goals, goal_imgs = [], [], []
    goal_imgs_processed = []
    for trip_fp in triple_filepaths:
        triple = np.load(
            os.path.join(data_path, trip_fp),
            allow_pickle=True
        )[()]
        start_img, lang_goal, goal_img = triple['start_img'], triple['lang_goal'], triple['goal_img']
        goal_img_processed = preprocess(Image.fromarray(goal_img.copy()))

        start_imgs.append(start_img)
        lang_goals.append(lang_goal)
        goal_imgs.append(goal_img)
        goal_imgs_processed.append(goal_img_processed)
    
    start_imgs = np.stack(start_imgs, axis=0)
    goal_imgs = np.stack(goal_imgs, axis=0)
    goal_imgs_processed = torch.stack(goal_imgs_processed, dim=0)
    return start_imgs, lang_goals, goal_imgs, goal_imgs_processed


def get_lang_goal_embeddings(model, lang_goal):
    with torch.no_grad():
        lang_goal_tokenized = clip.tokenize([lang_goal])
        lang_goal_features = model.encode_text(lang_goal_tokenized)
        lang_goal_features = lang_goal_features / lang_goal_features.norm(dim=1, keepdim=True)
    
    return lang_goal_features


def get_combined_goal_embeddings(model, preprocess, start_img, lang_goal):
    with torch.no_grad():
        start_img_processed = preprocess(Image.fromarray(start_img))
        lang_goal_tokenized = clip.tokenize([lang_goal])
        start_img_features = model.encode_image(start_img_processed.unsqueeze(0))
        lang_goal_features = model.encode_text(lang_goal_tokenized)
        combined_features = start_img_features + lang_goal_features
        combined_features = combined_features / combined_features.norm(dim=1, keepdim=True)

    return combined_features


def main():
    assert GOAL_TYPE in ['combined', 'lang']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device='cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)
    # if GOAL_TYPE == 'combined':
    #     model.load_state_dict(torch.load(MODEL_FP,  map_location=torch.device('cpu')))

    model.eval()

    start_imgs, lang_goals, goal_imgs, goal_imgs_processed = get_full_dataset(
        data_path=DATA_PATH, preprocess=preprocess
    )
    with torch.no_grad():
        goal_img_features = model.encode_image(goal_imgs_processed)
        goal_img_features = goal_img_features / goal_img_features.norm(dim=1, keepdim=True)

    goal_img_features = goal_img_features.cpu().numpy()
    assert len(goal_img_features.shape) == 2
    if OBJECTIVE == 'clip':
        knn = NearestNeighbors(algorithm='brute', metric='cosine').fit(goal_img_features)
    elif OBJECTIVE == 'L2':
        knn = NearestNeighbors(algorithm='brute').fit(goal_img_features)

    N = len(start_imgs)
    top_1_acc = 0
    top_K_acc = 0
    for i in range(N):
        print(f'Evaluating {i+1}/{N}')
        s = start_imgs[i]
        l = lang_goals[i]

        if GOAL_TYPE == 'combined':
            pred_goal_features = get_combined_goal_embeddings(model, preprocess, s, l)
        elif GOAL_TYPE == 'lang':
            pred_goal_features = get_lang_goal_embeddings(model, l)
        
        assert len(pred_goal_features.shape) == 2
        _, neighbor_idxs = knn.kneighbors(pred_goal_features, n_neighbors=K)
        neighbor_idxs = neighbor_idxs.squeeze()
        is_top_1 = int(i == neighbor_idxs[0])
        is_top_K = int(i in neighbor_idxs)

        top_1_acc += is_top_1
        top_K_acc += is_top_K

    top_1_acc /= N
    top_K_acc /= N

    print(f'Top 1 Acc: {top_1_acc}')
    print(f'Top K Acc: {top_K_acc}')


    # fig, axs = plt.subplots(N_QUERY, K+2, figsize=(8, 14))
    # fig.tight_layout(pad=2.0)
    # for n in range(N_QUERY):
    #     triple_fname = random.choice(os.listdir(DATA_PATH))
    #     triple = np.load(os.path.join(DATA_PATH, triple_fname), allow_pickle=True)[()]
    #     start_img, lang_goal, goal_img_gt = triple['start_img'], triple['lang_goal'], triple['goal_img']

    #     if GOAL_TYPE == 'combined':
    #         pred_goal_features = get_combined_goal_embeddings(model, preprocess, start_img, lang_goal)
    #     elif GOAL_TYPE == 'lang':
    #         pred_goal_features = get_lang_goal_embeddings(model, lang_goal)

    #     assert len(pred_goal_features.shape) == 2
    #     _, neighbor_idxs = knn.kneighbors(pred_goal_features, n_neighbors=K)
    #     neighbor_idxs = neighbor_idxs.squeeze()
    #     closest_goal_imgs = goal_imgs[neighbor_idxs]

    #     if GOAL_TYPE == 'combined':
    #         axs[n, 0].imshow(start_img)
    #     elif GOAL_TYPE == 'lang':
    #         axs[n, 0].imshow(np.zeros(start_img.shape, dtype=np.uint8))

    #     axs[n, 1].imshow(goal_img_gt)
    #     axs[n, 0].set_title('Start Img')
    #     axs[n, 1].set_title('Goal Img (GT)')
    #     axs[n, 0].get_xaxis().set_ticks([])
    #     axs[n, 0].get_yaxis().set_ticks([])
    #     axs[n, 1].get_xaxis().set_ticks([])
    #     axs[n, 1].get_yaxis().set_ticks([])
    #     axs[n, K].set_title(lang_goal + ' (Cols 3-5)')
    #     for k, closest_goal in enumerate(closest_goal_imgs):
    #         axs[n, k+2].imshow(closest_goal_imgs[k])
    #         axs[n, k+2].get_xaxis().set_ticks([])
    #         axs[n, k+2].get_yaxis().set_ticks([])

    # if GOAL_TYPE == 'combined':
    #     plt.savefig(f'visuals/combined_goal_neighbors_{SPLIT}_{OBJECTIVE}_{EXTRA}.png')
    # elif GOAL_TYPE == 'lang':
    #     plt.savefig(f'visuals/lang_goal_neighbors_{SPLIT}.png')






if __name__ == '__main__':
    main()