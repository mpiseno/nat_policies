from cliport.models.core.clip import load_clip
from nat_policies.models.core import build_model


def count_parameters(model, count_trainable_only=False):
    if count_trainable_only:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters())
        
    return num_params


def load_original_clip(variant, device='cpu'):
    model, _ = load_clip(variant, device=device)
    clip = build_model(model.state_dict()).to(device) # modifies CLIP model
    del model
    return clip    
    