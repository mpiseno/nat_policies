

def count_parameters(model, count_trainable_only=False):
    if count_trainable_only:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters())
        
    return num_params