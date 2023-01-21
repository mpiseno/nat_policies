import os

from cliport.models.core import fusion
from cliport.models.core.clip import CLIP, convert_weights
from cliport.models.core.fusion import FusionMult


class FusionMult_(FusionMult):
    def __init__(self, input_dim=3):
        super(FusionMult_, self).__init__(input_dim=input_dim)
    
    def tile_x2(self, x1, x2, x2_proj=None):
        if x2_proj:
            x2 = x2_proj(x2)

        x2 = x2.unsqueeze(-1).unsqueeze(-1)
        if x2.shape[0] == x1.shape[0]:
            x2 = x2.repeat(1, 1, x1.shape[-2], x1.shape[-1])
        else:
            x2 = x2.repeat(x1.shape[0], 1, x1.shape[-2], x1.shape[-1])
        
        return x2


fusion.names['mult_'] = FusionMult_


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
