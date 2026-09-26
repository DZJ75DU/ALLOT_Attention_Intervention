# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from subgraph_generator import (
    SubgraphGenerator,
    ExtractorSignals,
)

@dataclass
class ExtractorOutput:
    
    signals: ExtractorSignals
    aux_loss: Tensor
    node_emb: Optional[Tensor] = None

class ExtractorStrategy(nn.Module):

    def forward(
        self,
    ) -> ExtractorOutput:
        raise NotImplementedError


def _kl_to_bernoulli_prior(prob: Tensor, target_ratio: float, eps: float = 1e-6) -> Tensor:
    r = float(min(max(target_ratio, eps), 1.0 - eps))
    p = prob.clamp(min=eps, max=1.0 - eps)
    log_r = math.log(r)
    log_1_minus_r = math.log(1.0 - r)
    kl = p * (p.log() - log_r) + (1.0 - p) * ((1.0 - p).log() - log_1_minus_r)
    return kl.mean()


def _build_signals_from_weights(
    generator: SubgraphGenerator,
    batch: Data,
    pred_edge_weight: Tensor,
    edge_weight: Tensor,
) -> ExtractorSignals:

    node_batch = generator._resolve_node_batch(batch)
    causal_mask, spu_mask = generator.split_graph(
        batch=batch,
        pred_edge_weight=pred_edge_weight,
        node_batch=node_batch,
    )
    row = batch.edge_index[0]
    edge_batch = node_batch[row] if row.numel() > 0 else row.new_zeros(0, dtype=torch.long)
    return ExtractorSignals(
        edge_weight=edge_weight,
        causal_mask=causal_mask,
        spurious_mask=spu_mask,
        edge_batch=edge_batch,
        raw_edge_score=pred_edge_weight,
    )


class GSATExtractor(ExtractorStrategy):
    def __init__(
        self,
        target_ratio: float = 0.3,
        temperature: float = 1.0,
        kl_lambda: float = 1.0,
        node_emb_noise_sigma: float = 0.0,
    ):
        super().__init__()
        self.target_ratio = float(target_ratio)
        self.temperature = float(temperature)
        self.kl_lambda = float(kl_lambda)
        self.node_emb_noise_sigma = float(node_emb_noise_sigma)

    @staticmethod
    def _gumbel_sigmoid_sample(logits: Tensor, temperature: float, eps: float = 1e-6) -> Tensor:
        u = torch.rand_like(logits).clamp(min=eps, max=1.0 - eps)
        gumbel = torch.log(u) - torch.log1p(-u)
        return torch.sigmoid((logits + gumbel) / max(temperature, eps))

    def forward(
        self,
        generator: SubgraphGenerator,
        batch: Data,
        training: bool,
    ) -> ExtractorOutput:
        h, pred_edge_weight = generator.compute_edge_score(
            batch,
            node_emb_noise_sigma=self.node_emb_noise_sigma if training else 0.0,
        )

        if training:
            edge_weight = self._gumbel_sigmoid_sample(pred_edge_weight, self.temperature)
        else:
            edge_weight = torch.sigmoid(pred_edge_weight)

        signals = _build_signals_from_weights(
            generator=generator,
            batch=batch,
            pred_edge_weight=pred_edge_weight,
            edge_weight=edge_weight,
        )

        prob = torch.sigmoid(pred_edge_weight)
        aux_loss = self.kl_lambda * _kl_to_bernoulli_prior(prob, self.target_ratio)
        return ExtractorOutput(signals=signals, aux_loss=aux_loss, node_emb=h)


def build_extractor_strategy(name: str, args) -> ExtractorStrategy:

    return GSATExtractor(
        target_ratio=float(getattr(args, "target_ratio", 0.3)),
        temperature=float(getattr(args, "gsat_temperature", 1.0)),
        kl_lambda=float(getattr(args, "gsat_kl_lambda", 1.0)),
        node_emb_noise_sigma=0.0,
    )
