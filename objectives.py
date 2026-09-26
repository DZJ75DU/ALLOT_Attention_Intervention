# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@dataclass
class ObjectiveOutput:
    loss: Tensor 
    ce: Tensor 
    penalty: Tensor
    extra: Dict[str, float] = field(default_factory=dict)


def _ce(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits, labels.long())

class ObjectiveBase(nn.Module):
    name: str = "base"
    needs_env: bool = False

    def forward(
        self,
    ) -> ObjectiveOutput:
        raise NotImplementedError

    @staticmethod
    def _aux_or_zero(extractor_aux: Optional[Tensor], reference: Tensor) -> Tensor:
        if extractor_aux is None:
            return reference.new_zeros(())
        return extractor_aux.to(reference.device)


class ERMObjective(ObjectiveBase):
    name = "erm"
    needs_env = False

    def forward(self, *, logits, labels, extractor_aux=None):
        ce = _ce(logits, labels)
        aux = self._aux_or_zero(extractor_aux, ce)
        loss = ce + aux
        return ObjectiveOutput(
            loss=loss,
            ce=ce,
            penalty=ce.new_zeros(()),
            extra={"aux": float(aux.detach().item())},
        )

def build_objective(name: str, args) -> ObjectiveBase:

    return ERMObjective()
