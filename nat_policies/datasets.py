import os

from PIL import Image

import clip
import numpy as np

from torch.utils.data import Dataset


class CLIPortDataset(Dataset):
    def __init__(self, data_path, clip_preprocess=None):
        self.data_path = data_path
        self.clip_preprocess = clip_preprocess

        self.triple_filepaths = sorted(os.listdir(self.data_path))

    def __len__(self):
        return len(self.triple_filepaths)

    def __getitem__(self, index):
        triple = np.load(
            os.path.join(self.data_path, self.triple_filepaths[index]),
            allow_pickle=True
        )[()]
        start_img, lang_goal, goal_img = triple['start_img'], triple['lang_goal'], triple['goal_img']

        if self.clip_preprocess is not None:
            start_img = self.clip_preprocess(Image.fromarray(start_img))
            goal_img = self.clip_preprocess(Image.fromarray(goal_img))
            lang_goal = clip.tokenize([lang_goal])

        return (start_img, lang_goal, goal_img)
