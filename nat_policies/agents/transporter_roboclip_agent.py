import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor, Normalize, CenterCrop

import cliport.models as models
import cliport.models.core.fusion as fusion

from nat_policies.models.roboclip import RoboCLIP

from cliport.utils import utils
from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetTransporterAgent
from cliport.models.core.clip import build_model, load_clip, tokenize
from cliport.models.core import fusion
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up
from cliport.models.core.fusion import FusionConvLat
from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusionLat


class RoboCLIPAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        self.use_gt_goals = cfg['train']['use_gt_goals']
        super().__init__(name, cfg, train_ds, test_ds)

        print(f'Using GT Image Goals: {self.use_gt_goals}')

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = (
            'roboclip_lingunet_lat' if not self.use_gt_goals
            else 'roboclip_lingunet_lat_gt_vis'
        )
        self.attention = RoboCLIPAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = RoboCLIPTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type
        )
    
    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        goal = inp['goal_img'] if self.use_gt_goals else inp['lang_goal']

        out = self.attention.forward(inp_img, goal, softmax=softmax)
        return out

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        lang_goal = frame['lang_goal']
        goal_img = frame['goal_img']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal, 'goal_img': goal_img}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)
    
    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']
        goal = inp['goal_img'] if self.use_gt_goals else inp['lang_goal']

        out = self.transport.forward(inp_img, p0, goal, softmax=softmax)
        return out

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        lang_goal = frame['lang_goal']
        goal_img = frame['goal_img']

        inp = {'inp_img': inp_img, 'p0': p0, 'lang_goal': lang_goal, 'goal_img': goal_img}
        out = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta)
        return loss, err

    def act(self, obs, info, goal=None):
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)
        obs_goal, _, _, _ = goal
        img_goal = self.test_ds.get_image(obs_goal)
        lang_goal = info['lang_goal']

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal, 'goal_img': img_goal}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {
            'inp_img': img, 'p0': p0_pix,
            'lang_goal': lang_goal, 'goal_img': img_goal
        }
        place_conf = self.trans_forward(place_inp)
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': [p0_pix[0], p0_pix[1], p0_theta],
            'place': [p1_pix[0], p1_pix[1], p1_theta],
        }


class RoboCLIPAttention(TwoStreamAttentionLangFusionLat):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        self.use_gt_goals = cfg['train']['use_gt_goals']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")
    
    def forward(self, inp_img, goal, softmax=True):
        """Forward pass."""
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        for x in in_tens:
            lgts = self.attend(x, goal)
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output


class RoboCLIPTransport(TwoStreamTransportLangFusionLat):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        self.use_gt_goals = cfg['train']['use_gt_goals']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")


class RoboCLIPLingUNetLat(nn.Module):
    """
    CLIP RN50 with U-Net skip connections and lateral connections. This is almost identical to the
    CLIPLingUNetLat model from CLIPort, but CLIP is fine-tuned to enforce visual-language dynamics
    """

    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super(RoboCLIPLingUNetLat, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_dim = 2048  # penultimate layer channel-size of CLIP-RN50
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']
        self.goal_fusion_type = self.cfg['train']['goal_fusion_type']
        self.use_gt_goals = self.cfg['train']['use_gt_goals']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.preprocess = preprocess

        roboclip_ckpt_path = os.path.join(
            os.environ['NAT_POLICIES_ROOT'],
            cfg['train']['roboclip_ckpt_path']
        )
        clip_variant = 'RN50'
        self.roboclip = RoboCLIP(clip_variant=clip_variant, device=device)
        self.roboclip.inference_mode(ckpt_path=roboclip_ckpt_path)
        self._build_decoder()

        self.transporter_depth_mean = 0.00509261
        self.transporter_depth_std = 0.00903967
        self.transforms_goal = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.transforms_rgb = Compose([
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.padding = [(0, 0), (80, 80), (0, 0)]

    def _build_decoder(self):
        self.goal_fuser1 = fusion.names[self.goal_fusion_type](input_dim=self.input_dim // 2)
        self.goal_fuser2 = fusion.names[self.goal_fusion_type](input_dim=self.input_dim // 4)
        self.goal_fuser3 = fusion.names[self.goal_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 1024
        self.goal_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.goal_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.goal_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        self.up1 = Up(2048, 1024 // self.up_factor, self.bilinear)
        self.lat_fusion1 = FusionConvLat(input_dim=1024+512, output_dim=512)

        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear)
        self.lat_fusion2 = FusionConvLat(input_dim=512+256, output_dim=256)

        self.up3 = Up(512, 256 // self.up_factor, self.bilinear)
        self.lat_fusion3 = FusionConvLat(input_dim=256+128, output_dim=128)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion4 = FusionConvLat(input_dim=128+64, output_dim=64)

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion5 = FusionConvLat(input_dim=64+32, output_dim=32)

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion6 = FusionConvLat(input_dim=32+16, output_dim=16)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize([x]).to(self.device)
            text_feat, text_emb = self.clip.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def preprocess_rgb(self, img):
        # NOTE: unfortunately have to do this slow operation to convert each image to PIL image for preprocessing
        rgb_batch = img[:, :3]
        rgb_batch = rgb_batch.permute(0, 2, 3, 1).cpu().numpy()
        rgb_batch = [Image.fromarray(rgb.astype(np.uint8)) for rgb in rgb_batch]
        rgb_batch = [self.transforms_rgb(rgb).unsqueeze(0) for rgb in rgb_batch]
        rgb_batch = torch.cat(rgb_batch, dim=0).to(dtype=torch.float, device=self.device)
        return rgb_batch
    
    def compute_goal(self, goal):
        # Predict the goal
        assert(False, 'Fix the preprocessing here')
        rgb_tilde = x[:, :3].detach()
        rgb_tilde = tvF.resize(rgb_tilde, (224, 224))
        rgb_tilde = self.preprocess(rgb_tilde, dist='clip')
        with torch.no_grad():
            lang = tokenize([lang]).to(self.device)
            pred_goal, _, _ = self.roboclip(rgb_tilde, lang)
        
        return self._forward(x, lat, pred_goal)
    
    def _forward(self, rgb, lat, goal):
        x = rgb
        in_type = x.dtype
        in_shape = x.shape
        with torch.no_grad():
            pre_attnpool_acts, intermediates = self.roboclip.clip.visual.prepool_im(rgb)

        pre_attnpool_acts = pre_attnpool_acts.to(in_type)
        goal = goal.to(in_type)
        assert pre_attnpool_acts.shape[1] == self.input_dim

        x = self.conv1(pre_attnpool_acts)
        x = self.goal_fuser1(x, goal, x2_proj=self.goal_proj1)
        x = self.up1(x, intermediates[-2])
        x = self.lat_fusion1(x, lat[-6])

        x = self.goal_fuser2(x, goal, x2_proj=self.goal_proj2)
        x = self.up2(x, intermediates[-3])
        x = self.lat_fusion2(x, lat[-5])

        x = self.goal_fuser3(x, goal, x2_proj=self.goal_proj3)
        x = self.up3(x, intermediates[-4])
        x = self.lat_fusion3(x, lat[-4])

        x = self.layer1(x)
        x = self.lat_fusion4(x, lat[-3])

        x = self.layer2(x)
        x = self.lat_fusion5(x, lat[-2])

        x = self.layer3(x)
        x = self.lat_fusion6(x, lat[-1])

        x = self.conv2(x)

        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return x

    def forward(self, x, lat, goal):
        rgb = self.preprocess_rgb(x)
        goal_emb = self.compute_goal(goal)
        return self._forward(rgb, lat, goal_emb)


class RoboCLIPLingUNetLat_GTVis(RoboCLIPLingUNetLat):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super(RoboCLIPLingUNetLat_GTVis, self).__init__(
            input_shape, output_dim, cfg, device, preprocess
        )

    def preprocess_goal(self, goal):
        rgb_g = goal[..., :3]
        rgb_g = np.pad(rgb_g, self.padding, mode='constant')
        rgb_g = Image.fromarray(rgb_g.astype(np.uint8))
        rgb_g = self.transforms_goal(rgb_g)
        return rgb_g.unsqueeze(0).to(device=self.device)

    def compute_goal(self, goal):
        assert not isinstance(goal, str)
        goal = self.preprocess_goal(goal)
        goal = self.roboclip.encode_image(goal)
        return goal


models.names['roboclip_lingunet_lat'] = RoboCLIPLingUNetLat
models.names['roboclip_lingunet_lat_gt_vis'] = RoboCLIPLingUNetLat_GTVis
