"""Additional utilities for the main code.

You do *NOT* need to modify any code here.
"""

import enum

def update_target(model, target_model):
    """Copies the weight from one feedforward network to another.

    Args:
      model (nn.Module): a torch.nn.module instance
      target_model (nn.Module): a torch.nn.module instance
                                from the same parent as model
    """
    target_model.load_state_dict(model.state_dict())

class HERType(enum.Enum):
    NO_HINDSIGHT = 0
    FINAL = 1
    FUTURE = 2
    RANDOM = 3
