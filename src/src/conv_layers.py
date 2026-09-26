# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Optional, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv,
    BatchNorm,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)


GNNName = Literal[
    "gcn",
]

ActivationName = Literal[
    "relu",
]

PoolingName = Literal[
    "none",
]


@dataclass
class GNNOutput:

    node_emb: Tensor
    graph_emb: Optional[Tensor]
    layer_embs: List[Tensor]
    logits: Optional[Tensor]
    batch: Optional[Tensor]

def normalize_gnn_name(name: str) -> str:
    name = name.lower()
    table = {
        "gcn": "gcn",
    }
    if name not in table:
        raise ValueError(f"Unknown GNN type: {name}")
    return table[name]


def apply_activation(x: Tensor, activation: ActivationName) -> Tensor:
    if activation == "relu":
        return F.relu(x)
    raise ValueError(f"Unknown activation: {activation}")


def global_pool(
    x: Tensor,
    batch: Optional[Tensor],
    pooling: PoolingName = "mean",
) -> Optional[Tensor]:
    if pooling == "none":
        return None

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if pooling == "mean":
        return global_mean_pool(x, batch)
    if pooling in ("sum", "add"):
        return global_add_pool(x, batch)
    if pooling == "max":
        return global_max_pool(x, batch)

    raise ValueError(f"Unknown pooling: {pooling}")


def infer_batch_from_data(data: Data) -> Optional[Tensor]:
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch
    return None


class GNNLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        gnn_name: str = "gcn",
        edge_dim: Optional[int] = None,
        heads: int = 1,
        concat_heads: bool = False,
        batch_norm: bool = True,
        activation: ActivationName = "relu",
        residual: bool = True,
        dropout: float = 0.0,
        gin_mlp_layers: int = 2,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.gnn_name = normalize_gnn_name(gnn_name)
        self.edge_dim = edge_dim
        self.heads = int(heads)
        self.concat_heads = bool(concat_heads)

        self.use_batch_norm = bool(batch_norm)
        self.activation = activation
        self.residual = bool(residual)
        self.dropout = float(dropout)

        self.conv = self._build_conv(gin_mlp_layers=gin_mlp_layers)

        if self.use_batch_norm:
            self.norm = BatchNorm(self.output_dim)
        else:
            self.norm = None

        if self.residual and self.input_dim != self.output_dim:
            self.res_proj = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.res_proj = None

    def _build_conv(self, gin_mlp_layers: int) -> nn.Module:
        name = self.gnn_name

        if name == "gcn":
            return GCNConv(self.input_dim, self.output_dim)

        raise ValueError(f"Unknown GNN type: {name}")

    def _prepare_edge_weight(self, edge_attr: Optional[Tensor]) -> Optional[Tensor]:
        """
        Convert edge_attr to edge_weight for convs that expect scalar edge weights.
        """
        if edge_attr is None:
            return None

        if edge_attr.dim() == 1:
            return edge_attr.float()

        if edge_attr.dim() == 2 and edge_attr.size(-1) == 1:
            return edge_attr.view(-1).float()

        return None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        residual = x

        name = self.gnn_name

        if name in ("gcn"):
            edge_weight = self._prepare_edge_weight(edge_attr)
            h = self.conv(x, edge_index, edge_weight)

        else:
            h = self.conv(x, edge_index)

        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = apply_activation(h, self.activation)

        if self.residual:
            if self.res_proj is not None:
                residual = self.res_proj(residual)
            h = h + residual

        if self.norm is not None:
            if not (self.training and h.size(0) <= 1):
                h = self.norm(h)

        return h


class GNNEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        gnn_name: str = "gcn",
        edge_dim: Optional[int] = None,
        heads: int = 1,
        concat_heads: bool = False,
        batch_norm: bool = True,
        activation: ActivationName = "relu",
        residual: bool = True,
        dropout: float = 0.0,
        pooling: PoolingName = "mean",
        jk: Literal["last", "sum", "concat"] = "last",
    ):
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.gnn_name = normalize_gnn_name(gnn_name)
        self.pooling = pooling
        self.jk = jk

        layers = []
        for layer_idx in range(num_layers):
            in_dim = self.input_dim if layer_idx == 0 else self.hidden_dim
            use_residual = residual if layer_idx > 0 else False

            layers.append(
                GNNLayer(
                    input_dim=in_dim,
                    output_dim=self.hidden_dim,
                    gnn_name=self.gnn_name,
                    edge_dim=edge_dim,
                    heads=heads,
                    concat_heads=concat_heads,
                    batch_norm=batch_norm,
                    activation=activation,
                    residual=use_residual,
                    dropout=dropout,
                )
            )

        self.layers = nn.ModuleList(layers)

        if jk == "concat":
            self.output_dim = self.hidden_dim * self.num_layers
        else:
            self.output_dim = self.hidden_dim

    def forward(
        self,
        data_or_x: Union[Data, Tensor],
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> GNNOutput:

        if isinstance(data_or_x, Data):
            data = data_or_x
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
            batch = infer_batch_from_data(data)
        else:
            x = data_or_x
            if edge_index is None:
                raise ValueError("edge_index must be provided when input is Tensor.")

        if x is None:
            raise ValueError("Input node feature x is None.")

        if not x.is_floating_point():
            x = x.float()

        if edge_attr is not None and edge_attr.is_floating_point() is False:
            edge_attr = edge_attr.float()

        layer_embs: List[Tensor] = []
        h = x

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
            layer_embs.append(h)

        if self.jk == "last":
            node_emb = layer_embs[-1]
        elif self.jk == "sum":
            node_emb = torch.stack(layer_embs, dim=0).sum(dim=0)
        elif self.jk == "concat":
            node_emb = torch.cat(layer_embs, dim=-1)
        else:
            raise ValueError(f"Unknown jk mode: {self.jk}")

        graph_emb = global_pool(node_emb, batch, self.pooling)

        return GNNOutput(
            node_emb=node_emb,
            graph_emb=graph_emb,
            layer_embs=layer_embs,
            logits=None,
            batch=batch,
        )
