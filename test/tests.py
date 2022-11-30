import torch

from nat_policies.utils.eval_utils import cross_batch_L2


def test_cross_batch_L2():
    embeddings = torch.as_tensor([
        [0., 0., 0.],
        [0., 0., 0.]
    ])
    goal_embeddings = torch.as_tensor([
        [0., 0., 0.],
        [0., 0., 0.]
    ])
    metric = cross_batch_L2(embeddings, goal_embeddings)
    assert metric == 0.0

    embeddings = torch.as_tensor([
        [0., 0., 0.],
        [0., 0., 0.]
    ])
    goal_embeddings = torch.as_tensor([
        [0., 0., 0.],
        [1., 0., 0.]
    ])
    metric = cross_batch_L2(embeddings, goal_embeddings)
    assert metric == 0.5

    return True


def main():
    test_cross_batch_L2()

if __name__ == '__main__':
    main()