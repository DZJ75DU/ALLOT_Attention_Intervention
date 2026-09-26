# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from conv_layers import GNNEncoder


@dataclass
class SubgraphOutput:

    pred_edge_weight: Tensor
    causal_edge_mask: Tensor
    spu_edge_mask: Tensor
    edge_batch: Tensor
    node_emb: Tensor


@dataclass
class GraphPack:
    x: Tensor
    edge_index: Tensor
    edge_attr: Optional[Tensor]
    edge_weight: Tensor
    batch: Tensor


@dataclass
class BuiltSubgraphs:
    causal: GraphPack
    spurious: GraphPack


@dataclass
class ExtractorSignals:

    edge_weight: Tensor
    causal_mask: Tensor
    spurious_mask: Tensor
    edge_batch: Tensor
    raw_edge_score: Tensor


class SubgraphGenerator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        ratio: float = 0.3,
        gnn_type: str = "GCN",
    ):
        super().__init__()

        self.ratio = float(ratio)
        gnn_input_dim = input_dim

        gnn_edge_dim = None

        self.gnn_encoder = GNNEncoder(
            input_dim=gnn_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gnn_name=gnn_type,
            edge_dim=gnn_edge_dim,
            pooling="none",
        )

        self.edge_att = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, 1),
        )

    @staticmethod
    def _resolve_node_batch(batch: Data) -> Tensor:

        if hasattr(batch, "batch") and batch.batch is not None:
            return batch.batch
        return torch.zeros(
            batch.x.size(0),
            dtype=torch.long,
            device=batch.x.device,
        )

    def compute_edge_score(
        self,
        batch: Data,
        node_emb_noise_sigma: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:

        node_batch = self._resolve_node_batch(batch)

        edge_attr = None

        x_in = batch.x

        out = self.gnn_encoder(
            data_or_x=x_in,
            edge_index=batch.edge_index,
            edge_attr=edge_attr,
            batch=node_batch,
        )

        h = out.node_emb

        if node_emb_noise_sigma > 0.0 and self.training:
            h = h + torch.randn_like(h) * float(node_emb_noise_sigma)

        row, col = batch.edge_index
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        pred_edge_weight = self.edge_att(edge_rep).view(-1)

        return h, pred_edge_weight

    def graph_embedding(self, batch: Data) -> Tensor:
        from torch_geometric.nn import global_mean_pool
        h, _ = self.compute_edge_score(batch) 
        node_batch = self._resolve_node_batch(batch)
        return global_mean_pool(h, node_batch)

    @staticmethod
    def sparse_topk(
        src: Tensor,
        index: Tensor,
        ratio: float,
        descending: bool = True,
    ) -> Tuple[Tensor, Tensor]:

        device = src.device
        ratio = float(max(0.0, min(1.0, ratio)))

        reserve_mask = torch.zeros(src.size(0), dtype=torch.bool, device=device)

        for gid in torch.unique(index):
            edge_idx = torch.nonzero(index == gid, as_tuple=True)[0]
            if edge_idx.numel() == 0:
                continue

            k = int(torch.ceil(torch.tensor(edge_idx.numel() * ratio)).item())
            k = max(1, min(k, edge_idx.numel()))

            scores = src[edge_idx]
            selected_local = torch.topk(scores, k=k, largest=descending).indices
            selected_global = edge_idx[selected_local]

            reserve_mask[selected_global] = True

        new_idx_reserve = torch.nonzero(reserve_mask, as_tuple=True)[0]
        new_idx_drop = torch.nonzero(~reserve_mask, as_tuple=True)[0]

        return new_idx_reserve, new_idx_drop

    def split_graph(
        self,
        batch: Data,
        pred_edge_weight: Tensor,
        node_batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        if node_batch is None:
            node_batch = self._resolve_node_batch(batch)

        E = int(batch.edge_index.size(1))
        device = batch.edge_index.device

        if self.ratio < 0:
            causal_mask = torch.ones(E, dtype=torch.bool, device=device)
            spu_mask = torch.zeros(E, dtype=torch.bool, device=device)
            return causal_mask, spu_mask

        row = batch.edge_index[0]
        edge_batch = node_batch[row] if row.numel() > 0 else row.new_zeros(0, dtype=torch.long)

        new_idx_reserve, _ = self.sparse_topk(
            src=pred_edge_weight,
            index=edge_batch,
            ratio=self.ratio,
            descending=True,
        )

        causal_mask = torch.zeros(E, dtype=torch.bool, device=device)
        if new_idx_reserve.numel() > 0:
            causal_mask[new_idx_reserve] = True
        spu_mask = ~causal_mask

        return causal_mask, spu_mask

    def forward(self, batch: Data) -> SubgraphOutput:

        node_batch = self._resolve_node_batch(batch)

        h, pred_edge_weight = self.compute_edge_score(batch)
        causal_mask, spu_mask = self.split_graph(
            batch=batch,
            pred_edge_weight=pred_edge_weight,
            node_batch=node_batch,
        )

        row = batch.edge_index[0]
        edge_batch = (
            node_batch[row] if row.numel() > 0 else row.new_zeros(0, dtype=torch.long)
        )

        return SubgraphOutput(
            pred_edge_weight=pred_edge_weight,
            causal_edge_mask=causal_mask,
            spu_edge_mask=spu_mask,
            edge_batch=edge_batch,
            node_emb=h,
        )
    @staticmethod
    def _build_subgraph(
        x_source: Tensor,
        batch: Data,
        edge_indices: Tensor,
        edge_weight: Tensor,
    ) -> GraphPack:
        device = batch.edge_index.device

        selected_edge_index = batch.edge_index[:, edge_indices]
        edge_attr = getattr(batch, "edge_attr", None)
        selected_edge_attr = edge_attr[edge_indices] if edge_attr is not None else None
        selected_edge_weight = edge_weight[edge_indices]

        if selected_edge_index.numel() == 0:
            node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=device)
            node_mask[0] = True
        else:
            node_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=device)
            node_mask[selected_edge_index.reshape(-1)] = True

        selected_nodes = torch.nonzero(node_mask, as_tuple=True)[0]

        sub_edge_index, sub_edge_attr = subgraph(
            subset=selected_nodes,
            edge_index=selected_edge_index,
            edge_attr=selected_edge_attr,
            relabel_nodes=True,
            num_nodes=batch.x.size(0),
        )

        sub_x = x_source[selected_nodes]

        if hasattr(batch, "batch") and batch.batch is not None:
            sub_batch = batch.batch[selected_nodes]
        else:
            sub_batch = torch.zeros(sub_x.size(0), dtype=torch.long, device=device)

        assert selected_edge_weight.numel() == sub_edge_index.size(1), (
            f"edge_weight mismatch: weight={selected_edge_weight.numel()}, "
            f"edge_index={sub_edge_index.size(1)}"
        )

        return GraphPack(
            x=sub_x,
            edge_index=sub_edge_index,
            edge_attr=sub_edge_attr,
            edge_weight=selected_edge_weight,
            batch=sub_batch,
        )

    def build_pyg_subgraphs(
        self,
        out: SubgraphOutput,
        batch: Data,
    ) -> BuiltSubgraphs:

        x_source = batch.x

        causal_idx = torch.nonzero(out.causal_edge_mask, as_tuple=True)[0]
        spu_idx = torch.nonzero(out.spu_edge_mask, as_tuple=True)[0]

        causal_pack = self._build_subgraph(
            x_source=x_source,
            batch=batch,
            edge_indices=causal_idx,
            edge_weight=out.pred_edge_weight,
        )
        spu_pack = self._build_subgraph(
            x_source=x_source,
            batch=batch,
            edge_indices=spu_idx,
            edge_weight=-out.pred_edge_weight,
        )

        return BuiltSubgraphs(causal=causal_pack, spurious=spu_pack)
