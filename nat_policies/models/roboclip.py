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

from nat_policies.utils.common import load_original_clip


class RoboCLIP(nn.Module):
    def __init__(self, clip_variant, device='cpu'):
        super(RoboCLIP, self).__init__()
        assert clip_variant in ['RN50', 'ViT']

        self.clip_variant = 'RN50' if clip_variant == 'RN50' else 'ViT-B/32'
        self.clip_finetuned_layers = None
        if self.clip_variant == 'RN50':
            self.clip_finetuned_layers = [
                'visual.attnpool', # Visual encoder
                'ln_final', 'transformer.resblocks.11', # Language encoder
            ]
        elif self.clip_variant == 'ViT-B/32':
            self.clip_finetuned_layers = [
                'ln_post', 'visual.transformer.resblocks.11', # Visual encoder
                'ln_final', 'transformer.resblocks.11', # Language encoder
            ]

        self.device = device
        self.trained_layers = []
        self._build_model()

    def _build_model(self):
        # Loads the CLIP model from original CLIP repo
        self.clip = load_original_clip(variant=self.clip_variant, device=self.device)
        self._freeze_clip_layers(self.clip)

        self.clip_embed_dim = self.clip.text_projection.shape[1]
        film_dim = 512
        self.film_generator = nn.Sequential(
            nn.Linear(self.clip_embed_dim, film_dim),
            nn.GELU(),
            nn.Linear(film_dim, film_dim),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(film_dim, self.clip_embed_dim)
        self.beta_head = nn.Linear(film_dim, self.clip_embed_dim)
    
        self.logit_scale = nn.Parameter(self.clip.logit_scale.clone(), requires_grad=True)
        self.trained_layers.append('fusion')
        #print(f'Trained layers: {self.trained_layers}')
    
    def _freeze_clip_layers(self, clip_model):
        for name, param in clip_model.named_parameters():
            param.requires_grad = False
            if any([name.startswith(layer) for layer in self.clip_finetuned_layers]):
                param.requires_grad = True
                self.trained_layers.append(name)

        # Make sure to handle unnamed parameters!!!
        clip_model.logit_scale.requires_grad = True
        clip_model.text_projection.requires_grad = True
        self.trained_layers.append('logit_scale')
        self.trained_layers.append('text_projection')
        if self.clip_variant == 'ViT-B/32':
            clip_model.visual.proj.requires_grad = True
            self.trained_layers.append('visual_projection')
    
    def inference_mode(self, ckpt_path):
        def modify_state_dict(state_dict):
            return {key[len('roboclip.'):]: val for key, val in state_dict.items()}

        ckpt = torch.load(ckpt_path)
        state_dict = modify_state_dict(ckpt['state_dict'])
        self.load_state_dict(state_dict)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        
        print(f'Loaded RoboCLIP model from: {ckpt_path}')
        
    def encode_image(self, img):
        return self.clip.encode_image(img)

    def encode_text(self, lang):
        return self.clip.encode_text(lang)

    def estimate_goal(self, img_emb, lang_emb):
        in_dtype = img_emb.dtype
        x = lang_emb.to(torch.float32)
        x = self.film_generator(x)  
        gamma = self.gamma_head(x).to(in_dtype)
        beta = self.beta_head(x).to(in_dtype)
        pred_goal = gamma * img_emb + beta
        return pred_goal
        
    def forward(self, img, lang):
        print(f'Batch size: {img.shape[0]}')
        img_embedding = self.encode_image(img)
        lang_embedding = self.encode_text(lang)
        pred_goal_embedding = self.estimate_goal(img_embedding, lang_embedding)
        return pred_goal_embedding, img_embedding, lang_embedding
