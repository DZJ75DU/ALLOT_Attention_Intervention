# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ClassifierOutput:
    logits: Tensor
    prob: Tensor
    pred: Tensor
    loss: Optional[Tensor] = None


class LinearClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        if use_layer_norm:
            self.norm = nn.LayerNorm(input_dim)
        else:
            self.norm = nn.Identity()

        self.fc = nn.Linear(input_dim, num_classes)

    def forward(
        self,
        graph_emb: Tensor,
        labels: Optional[Tensor] = None,
    ) -> ClassifierOutput:
        graph_emb = graph_emb.to(dtype=self.fc.weight.dtype)
        graph_emb = self.norm(graph_emb)
        logits = self.fc(graph_emb)

        prob = F.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device).long()
            loss = F.cross_entropy(logits, labels)

        return ClassifierOutput(
            logits=logits,
            prob=prob,
            pred=pred,
            loss=loss,
        )


def build_classifier(
    classifier_type: str,
    input_dim: int,
    num_classes: int,
    use_layer_norm: bool = True,
) -> nn.Module:

    classifier_type = classifier_type.lower()
    if classifier_type == "linear":
        return LinearClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            use_layer_norm=use_layer_norm,
        )

    raise ValueError(f"Unknown classifier_type: {classifier_type}")