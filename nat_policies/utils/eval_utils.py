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


def ground_truth_cossim(Z_s, Z_g):
    sim_matrix = Z_s @ Z_g.t()
    return torch.mean(torch.diag(sim_matrix))


def cross_batch_cossim(Z_s, Z_g):
    sim_matrix = Z_s @ Z_g.t()
    all_sim_vals = torch.sum(sim_matrix)
    diag_sim_vals = torch.sum(torch.diag(sim_matrix))
    off_diag_sim_sum = all_sim_vals - diag_sim_vals

    B = Z_s.shape[0]
    num_off_diag = B * B - B
    return off_diag_sim_sum / num_off_diag


def knn_classification(embeddings, goal_embeddings, K=5, N=30):
    def get_knn(sample, metric):
        knn = NearestNeighbors(algorithm='brute', metric=metric).fit(goal_embeddings)
        _, nn_idxs = knn.kneighbors(sample, n_neighbors=K)
        return nn_idxs

    embeddings = embeddings.cpu().numpy()
    goal_embeddings = goal_embeddings.cpu().numpy()

    sample_idxs = np.random.choice(len(embeddings), size=N)
    sample = embeddings[sample_idxs]
    neighbor_idxs_l2 = get_knn(sample, metric='minkowski')
    neighbor_idxs_cosine = get_knn(sample, metric='cosine')
    
    top_1_acc_l2, top_5_acc_l2 = 0, 0
    top_1_acc_cosine, top_5_acc_cosine = 0, 0
    for i in range(N):
        sample_i = sample_idxs[i]
        top_1_acc_l2 += int(sample_i == neighbor_idxs_l2[i][0])
        top_5_acc_l2 += int(sample_i in neighbor_idxs_l2[i])
        top_1_acc_cosine += int(sample_i == neighbor_idxs_cosine[i][0])
        top_5_acc_cosine += int(sample_i in neighbor_idxs_cosine[i])

    top_1_acc_l2 /= N
    top_5_acc_l2 /= N
    top_1_acc_cosine /= N
    top_5_acc_cosine /= N
    return top_1_acc_l2, top_5_acc_l2, top_1_acc_cosine, top_5_acc_cosine


def start_pred_goal_ratio(start, pred, goal):
    pred_goal_dist = torch.linalg.norm(goal - pred, ord=2, dim=-1)
    start_goal_dist = torch.linalg.norm(goal - start, ord=2, dim=-1)
    pred_goal_ratio = torch.mean(pred_goal_dist / (start_goal_dist + 1e-6))
    return pred_goal_ratio
