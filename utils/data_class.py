import torch
from typing import NamedTuple, Optional


class PVTrainDataSet(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: Optional[torch.Tensor]


class PVTestDataSet(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: Optional[torch.Tensor]
