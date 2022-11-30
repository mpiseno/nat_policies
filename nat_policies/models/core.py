import os
import torch
import torch.nn as nn

from cliport.models.core.clip import CLIP, convert_weights, load_clip


class RoboCLIP(nn.Module):
    def __init__(self, clip_variant, LP_phase=False, device='cpu'):
        super(RoboCLIP, self).__init__()
        assert clip_variant in ['RN50', 'ViT']
        self.clip_variant = 'RN50' if clip_variant == 'RN50' else 'ViT-B/32'
        self.LP_phase = LP_phase
        self.clip_finetuned_layers = []
        if self.LP_phase:
            if self.clip_variant == 'RN50':
                raise Exception() # TODO: fill this out
                self.clip_finetuned_layers = []
            elif self.clip_variant == 'ViT-B/32':
                # LP phase needs to also modify vision layers
                self.clip_finetuned_layers = [
                    'visual.transformer.resblocks.11',
                    'visual.ln_post'
                ]
        else:
            if self.clip_variant == 'RN50':
                raise Exception() # TODO: fill this out
                self.clip_finetuned_layers = []
            elif self.clip_variant == 'ViT-B/32':
                # FT phase needs to finetune both vision and lang
                self.clip_finetuned_layers = [
                    'visual.transformer.resblocks.11', 'visual.ln_post',  # Vision layers
                    'transformer.resblocks.11',                 # Lang layers
                    'ln_final'
                ]

        self.device = device
        self._build_model()

    def _freeze_clip_layers(self, clip_model):
        trained_layers = []
        for name, param in clip_model.named_parameters():
            param.requires_grad = False
            if any([name.startswith(layer) for layer in self.clip_finetuned_layers]):
                param.requires_grad = True
                trained_layers.append(name)

        # Make sure to handle unnamed parameters!!!
        clip_model.logit_scale.requires_grad = False
        if self.clip_variant == 'ViT-B/32':
            clip_model.visual.proj.requires_grad = True
            trained_layers.append('visual_projection')

        if self.LP_phase:
            clip_model.text_projection.requires_grad = False
        else:
            clip_model.text_projection.requires_grad = True
            trained_layers.append('text_projection')
        
        return trained_layers

    def _build_model(self):
        model, _ = load_clip(self.clip_variant, device=self.device) # Loads the CLIP model from original CLIP repo
        clip = build_model(model.state_dict()).to(self.device) # modifies CLIP model
        del model
        
        self.clip = clip
        trained_layers = self._freeze_clip_layers(self.clip)

        self.clip_embed_dim = self.clip.text_projection.shape[1]
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.clip_embed_dim, self.clip_embed_dim),
            nn.ReLU(),
            nn.Linear(self.clip_embed_dim, self.clip_embed_dim)
        )
        self.logit_scale = nn.Parameter(self.clip.logit_scale.clone(), requires_grad=True)

        trained_layers.extend(['logit_scale', 'fusion'])
        print(f'RoboCLIP trained layers: {trained_layers}')
        
    def encode_image(self, img):
        return self.clip.encode_image(img)

    def encode_text(self, lang):
        return self.clip.encode_text(lang)

    def forward(self, img, lang):
        img_embedding = self.encode_image(img)
        lang_embedding = self.encode_text(lang)
        in_dtype = img_embedding.dtype

        fusion_input = torch.cat((img_embedding, lang_embedding), dim=-1).to(torch.float32)
        fused_embedding = self.fusion(fusion_input).to(in_dtype)
        return fused_embedding, img_embedding, lang_embedding


def build_model(state_dict: dict):
    '''
    Adapted from CLIPort code. Use this method to load CLIP model for fine-tuning and for use in
    a CLIPort-like architecture (e.g. for RoboCLIPLingUNetLat).

    state_dict is the state_dict from the original CLIP repo. We will load in a different state_dict
    that has our fine-tuned weights.
    '''
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # raw_layers = ["visual.attnpool"]
    # for layer in raw_layers:
    #     del_keys = []
    #     for k in state_dict.keys():
    #         if k.startswith(layer):
    #             del_keys.append(k)
        
    #     for k in del_keys:
    #         del state_dict[k]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()