import os

import numpy as np
import matplotlib.pyplot as plt

from nat_policies.datasets import RavensDataset
from cliport.utils import utils


def main():
    task = 'put-block-in-bowl-seen-colors'
    mode = 'train'

    # Load configs
    root_dir = os.environ['NAT_POLICIES_ROOT']
    cliport_root_dir = os.environ['CLIPORT_ROOT']
    config_file = 'train.yaml' 
    cfg = utils.load_hydra_config(os.path.join(root_dir, f'nat_policies/cfg/{config_file}'))

    # Override defaults
    cfg['task'] = task
    cfg['mode'] = mode

    data_dir = os.path.join(cliport_root_dir, 'data')
    train_ds = RavensDataset(os.path.join(data_dir, f'{task}-train'), cfg, n_demos=100, augment=True)

    sample, goal = train_ds[0]

    '''
    pv = (320, 160) // 2 = (160, 80)

    # do 1 rotation
    '''


    # N = 4

    # fig, axs = plt.subplots(N, 2, figsize=(8, 8))
    # fig.tight_layout(pad=2.0)
    # for i, triple_filepath in enumerate(os.listdir(datapath)[:N]):
    #     triple_filepath = os.path.join(datapath, triple_filepath)

    #     triple = np.load(triple_filepath, allow_pickle=True)[()]
    #     axs[i, 0].imshow(triple['start_img'])
    #     axs[i, 1].imshow(triple['goal_img'])
    #     axs[i, 1].set_title(triple['lang_goal'])

    #     axs[i, 0].get_xaxis().set_ticks([])
    #     axs[i, 0].get_yaxis().set_ticks([])
    #     axs[i, 1].get_xaxis().set_ticks([])
    #     axs[i, 1].get_yaxis().set_ticks([])

    # plt.savefig('triple_vis.png')


if __name__ == '__main__':
    main()

