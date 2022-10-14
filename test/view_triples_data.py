import os
from re import L

import numpy as np
import matplotlib.pyplot as plt


def main():
    datapath = 'data/put-block-in-bowl-seen-colors-train'
    N = 4

    fig, axs = plt.subplots(N, 2, figsize=(8, 8))
    fig.tight_layout(pad=2.0)
    for i, triple_filepath in enumerate(os.listdir(datapath)[:N]):
        triple_filepath = os.path.join(datapath, triple_filepath)

        triple = np.load(triple_filepath, allow_pickle=True)[()]
        axs[i, 0].imshow(triple['start_img'])
        axs[i, 1].imshow(triple['goal_img'])
        axs[i, 1].set_title(triple['lang_goal'])

        axs[i, 0].get_xaxis().set_ticks([])
        axs[i, 0].get_yaxis().set_ticks([])
        axs[i, 1].get_xaxis().set_ticks([])
        axs[i, 1].get_yaxis().set_ticks([])

    plt.show()


if __name__ == '__main__':
    main()

