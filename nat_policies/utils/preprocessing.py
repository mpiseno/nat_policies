import torch
import numpy as np


def preprocess_rgb(img, dist):
    """Pre-process input (subtract mean, divide by std)."""

    transporter_color_mean = [0.18877631, 0.18877631, 0.18877631]
    transporter_color_std = [0.07276466, 0.07276466, 0.07276466]
    transporter_depth_mean = 0.00509261
    transporter_depth_std = 0.00903967

    franka_color_mean = [0.622291933, 0.628313992, 0.623031488]
    franka_color_std = [0.168154213, 0.17626014, 0.184527364]
    franka_depth_mean = 0.872146842
    franka_depth_std = 0.195743116

    clip_color_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_color_std = [0.26862954, 0.26130258, 0.27577711]

    # choose distribution
    if dist == 'clip':
        color_mean = clip_color_mean
        color_std = clip_color_std
    elif dist == 'franka':
        color_mean = franka_color_mean
        color_std = franka_color_std
    else:
        color_mean = transporter_color_mean
        color_std = transporter_color_std

    if dist == 'franka':
        depth_mean = franka_depth_mean
        depth_std = franka_depth_std
    else:
        depth_mean = transporter_depth_mean
        depth_std = transporter_depth_std

    assert isinstance(img, torch.Tensor)
    def cast_shape(stat, img):
        tensor = torch.from_numpy(np.array(stat)).to(device=img.device, dtype=img.dtype)
        tensor = tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        tensor = tensor.repeat(img.shape[0], 1, img.shape[-2], img.shape[-1])
        return tensor

    color_mean = cast_shape(color_mean, img)
    color_std = cast_shape(color_std, img)
    depth_mean = cast_shape(depth_mean, img)
    depth_std = cast_shape(depth_std, img)

    # normalize
    img = img.clone()
    img[:, :3, :, :] = ((img[:, :3, :, :] / 255 - color_mean) / color_std)
    img[:, 3:, :, :] = ((img[:, 3:, :, :] - depth_mean) / depth_std)

    return img