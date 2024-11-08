import torch

from models.zero_shot_models.utils.fc_out_model import FcOutModel


class MscnAggregator(FcOutModel):
    """
    Class used only for msg aggregation as part of topological mp module
    """

    def __init__(self, hidden_dim=0, test: bool = False, **kwargs):
        super().__init__(input_dim=2 * hidden_dim, output_dim=hidden_dim, **kwargs)
        self.test = test

    def forward(self, updates, previous_state):
        if not self.test:
            new_state = self.fcout(torch.cat([updates, previous_state], dim=1))
        else:
            new_state = torch.add(updates, previous_state)
        return new_state
