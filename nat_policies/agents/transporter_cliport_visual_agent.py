import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cliport.models as models

from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetTransporterAgent
from nat_policies.utils.preprocessing import preprocess_rgb

from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusionLat
from cliport.models.core.clip import build_model, load_clip
from cliport.models.core import fusion
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up
from cliport.models.core.fusion import FusionConvLat


class CLIPortVisualGoalAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'cliport_visual_lingunet_let'
        self.attention = TwoStreamAttentionVisualGoalFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=preprocess_rgb,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportVisualGoalFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=preprocess_rgb,
            cfg=self.cfg,
            device=self.device_type
        )

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        goal = inp['goal_img']

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
        goal = inp['goal_img']

        out = self.transport.forward(inp_img, p0, goal, softmax=softmax)
        return out

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        goal_img = frame['goal_img']

        inp = {'inp_img': inp_img, 'p0': p0, 'goal_img': goal_img}
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



class TwoStreamAttentionVisualGoalFusionLat(TwoStreamAttentionLangFusionLat):
    """Two Stream Visual Goal-Conditioned Attention (a.k.a Pick) module."""

    def __init__(
        self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device
    ):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")


class TwoStreamTransportVisualGoalFusionLat(TwoStreamTransportLangFusionLat):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(
        self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device
    ):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
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


class CLIPVisualLingUNetLat(nn.Module):
    """ CLIP RN50 with U-Net skip connections and lateral connections """

    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super(CLIPVisualLingUNetLat, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_dim = 2048  # penultimate layer channel-size of CLIP-RN50
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']
        self.goal_fusion_type = self.cfg['train']['goal_fusion_type']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.preprocess = preprocess

        self._load_clip()
        self._build_decoder()

        self.padding = [(0, 0), (80, 80), (0, 0)]

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)
        del model

    def _build_decoder(self):
        # language
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

    def encode_image(self, img):
        with torch.no_grad():
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)

        return img_encoding, img_im

    def encode_image_goal(self, goal_img):
        with torch.no_grad():
            img_encoding = self.clip_rn50.visual(goal_img)

        return img_encoding

    def preprocess_goal(self, goal):
        img_unprocessed_goal = np.pad(goal, self.padding, mode='constant')
        img_unprocessed_goal = np.resize(img_unprocessed_goal, (224, 224, 6))
        input_data_goal = img_unprocessed_goal
        in_shape_goal = (1,) + input_data_goal.shape
        input_data_goal = input_data_goal.reshape(in_shape_goal)
        in_tensor_goal = torch.from_numpy(input_data_goal).to(
            dtype=torch.float, device=self.device
        ).permute(0, 3, 1, 2)
        return in_tensor_goal

    def forward(self, x, lat, g):
        x = self.preprocess(x, dist='clip')
        g = self.preprocess_goal(g)
        g = self.preprocess(g, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  # select RGB
        g = g[:,:3]

        x, im = self.encode_image(x)
        x = x.to(in_type)

        g = self.encode_image_goal(g)
        g = g.to(in_type)
        
        assert x.shape[1] == self.input_dim
        x = self.conv1(x)

        x = self.goal_fuser1(x, g, x2_proj=self.goal_proj1)
        x = self.up1(x, im[-2])
        x = self.lat_fusion1(x, lat[-6])

        x = self.goal_fuser2(x, g, x2_proj=self.goal_proj2)
        x = self.up2(x, im[-3])
        x = self.lat_fusion2(x, lat[-5])

        x = self.goal_fuser3(x, g, x2_proj=self.goal_proj3)
        x = self.up3(x, im[-4])
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


models.names['cliport_visual_lingunet_let'] = CLIPVisualLingUNetLat