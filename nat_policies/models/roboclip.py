import torch
import torch.nn as nn
import torch.nn.functional as F

import cliport.utils.utils as utils
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up
from cliport.models.core.clip import tokenize
from cliport.models.core import fusion
from cliport.models.core.fusion import FusionConvLat
from cliport.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusion
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusion
from cliport.models.resnet_lat import ResNet45_10s

from nat_policies.models.clip import build_model, RoboCLIP


class TwoStreamAttentionLangFusionLat(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = 'plain_resnet_lat', 'roboclip_lingunet_lat'
        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

        self.attn_steam_one = ResNet45_10s(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = RoboCLIPLingUNetLat(self.in_shape, 1, self.cfg, self.device, self.preprocess)

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)
        return x


class TwoStreamTransportLangFusionLat(TwoStreamTransportLangFusion):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = 'plain_resnet_lat', 'roboclip_lingunet_lat'
        stream_one_model = ResNet45_10s
        stream_two_model = RoboCLIPLingUNetLat

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel


class RoboCLIPLingUNetLat(RoboCLIP):
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
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.preprocess = preprocess

        finetuned_clip_dir = cfg['train']['finetuned_clip_dir']
        self.clip_rn50 = RoboCLIP._load_clip(self.device, finetuned_clip_dir)
        self._build_decoder()

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

    def encode_image(self, img):
        with torch.no_grad():
            pre_attnpool_activations, intermediates = self.clip_rn50.visual.prepool_im(img)
            visual_embedding = self.clip_rn50.visual.attnpool(pre_attnpool_activations)

        return pre_attnpool_activations, intermediates, visual_embedding

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize([x]).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def forward(self, x, lat, l):
        x = self.preprocess(x, dist='clip')

        # TODO: visualize the output of preprocessing and make sure finetuning does the same thing
        import pdb; pdb.set_trace()

        in_type = x.dtype
        in_shape = x.shape

        # Handle visual
        rgb = x[:, :3]  # select RGB
        pre_attnpool_activations, intermediates, visual_embedding = self.encode_image(rgb)
        pre_attnpool_activations = pre_attnpool_activations.to(in_type)
        assert pre_attnpool_activations.shape[1] == self.input_dim

        # Handle goal embedding
        # NOTE: I use 'lang_embedding' where cliport uses 'l_enc'
        lang_embedding, l_emb, l_mask = self.encode_text(l)
        lang_embedding = lang_embedding.to(dtype=pre_attnpool_activations.dtype)
        goal_embedding = visual_embedding + lang_embedding

        x = self.conv1(pre_attnpool_activations)

        x = self.goal_fuser1(x, goal_embedding, x2_proj=self.goal_proj1)
        x = self.up1(x, intermediates[-2])
        x = self.lat_fusion1(x, lat[-6])

        x = self.goal_fuser2(x, goal_embedding, x2_proj=self.goal_proj2)
        x = self.up2(x, intermediates[-3])
        x = self.lat_fusion2(x, lat[-5])

        x = self.goal_fuser3(x, goal_embedding, x2_proj=self.goal_proj3)
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