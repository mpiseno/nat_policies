import torch
import numpy as np

from sklearn.neighbors import NearestNeighbors


def cross_batch_L2(embeddings, goal_embeddings):
    cross_batch_L2_dist = 0
    for i, z in enumerate(embeddings):
        dist = torch.linalg.norm(goal_embeddings - z, ord=2, dim=-1)
        dist = dist[torch.arange(len(embeddings)) != i]
        avg_dist = torch.mean(dist)
        cross_batch_L2_dist += avg_dist
        
    cross_batch_L2_dist /= len(embeddings)
    return cross_batch_L2_dist.item()


def ground_truth_L2(embeddings, goal_embeddings):
    l2_dist = torch.linalg.norm(goal_embeddings - embeddings, ord=2, dim=-1)
    avg_l2_dist = torch.mean(l2_dist)
    return avg_l2_dist.item()


def knn_classification(embeddings, goal_embeddings, K=5, N=30):
    embeddings = embeddings.cpu().numpy()
    goal_embeddings = goal_embeddings.cpu().numpy()

    knn = NearestNeighbors(algorithm='brute').fit(goal_embeddings)
    sample_idxs = np.random.choice(len(embeddings), size=N)
    sample = embeddings[sample_idxs]
    _, neighbor_idxs = knn.kneighbors(sample, n_neighbors=K)
    top_1_acc, top_5_acc = 0, 0
    for i in range(N):
        sample_i = sample_idxs[i]
        top_1_acc += int(sample_i == neighbor_idxs[i][0])
        top_5_acc += int(sample_i in neighbor_idxs[i])

    top_1_acc /= N
    top_5_acc /= N
    return top_1_acc, top_5_acc


def start_pred_goal_ratio(start, pred, goal):
    pred_goal_dist = torch.linalg.norm(goal - pred, ord=2, dim=-1)
    start_goal_dist = torch.linalg.norm(goal - start, ord=2, dim=-1)
    pred_goal_ratio = torch.mean(pred_goal_dist / (start_goal_dist + 1e-6))
    return pred_goal_ratio
