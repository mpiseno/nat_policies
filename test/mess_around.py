from nat_policies.models.core import RoboCLIP
from nat_policies.utils.common import count_parameters


def get_trainable_param_names(model):
    return [name for name, p in model.named_parameters() if p.requires_grad]


model = RoboCLIP('ViT', finetune_clip_layers=False)
print(get_trainable_param_names(model))
n_trainable_params = count_parameters(model, count_trainable_only=True)

import pdb; pdb.set_trace()
print(n_trainable_params)
