import os

import numpy as np



def main():
    cliport_datapath = '../cliport/data/put-block-in-bowl-seen-colors-train/'

    triples = []
    cliport_rgb_path = os.path.join(cliport_datapath, 'color')
    cliport_info_path = os.path.join(cliport_datapath, 'info')
    for episode_file in sorted(os.listdir(cliport_rgb_path)):
        rgb_episode_path = os.path.join(cliport_rgb_path, episode_file)
        info_episode_path = os.path.join(cliport_info_path, episode_file)

        rgb_episode = np.load(rgb_episode_path, allow_pickle=True)
        info_episode = np.load(info_episode_path, allow_pickle=True)

        camera_view_idx = 0     # CLIP episode arrays have multiple cam views
        for t in range(rgb_episode.shape[0] - 1):
            start_img = rgb_episode[t, camera_view_idx].copy()
            lang_goal = info_episode[t]['lang_goal']
            goal_img = rgb_episode[t+1, camera_view_idx].copy()
            triples.append({
                'start_img': start_img, 
                'lang_goal': lang_goal,
                'goal_img': goal_img
            })

    our_datapath = 'data/put-block-in-bowl-seen-colors-train/'
    for i, trip in enumerate(triples):
        trip_filepath = os.path.join(our_datapath, f'triple_{i}.npy')
        np.save(trip_filepath, trip, allow_pickle=True)


if __name__ == '__main__':
    main()