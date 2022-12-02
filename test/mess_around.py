from nat_policies.models.core import RoboCLIP
from nat_policies.utils.common import count_parameters


def get_trainable_param_names(model):
    return [name for name, p in model.named_parameters() if p.requires_grad]


