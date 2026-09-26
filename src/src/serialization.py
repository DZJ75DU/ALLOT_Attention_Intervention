# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import Tensor
from torch_geometric.data import Data

@dataclass
class SerializedGraph:
    prompt: str
    edge_text: str
    edge_spans: List[Dict]
    num_nodes: int
    num_edges: int
    graph_id: Optional[int] = None


@dataclass
class GraphSerializerOutput:
    prompts: List[str]
    records: List[SerializedGraph]
    input_ids: Tensor
    attention_mask: Tensor
    edge_token_matrix: Tensor
    graph_token_mask: Tensor
    edge_score: Optional[Tensor]
    edge_score_mask: Tensor
    edge_indices: List[Tensor]


def build_edge_text_with_spans(
    edges: List[Tuple[int, int]],
) -> Tuple[str, List[Dict]]:
    parts = ["["]
    spans = []

    for i, (u, v) in enumerate(edges):
        if i > 0:
            parts.append(", ")

        start = sum(len(p) for p in parts)
        edge_str = f"({u}, {v})"
        parts.append(edge_str)
        end = start + len(edge_str)

        spans.append({
            "edge_id": int(i),
            "u": int(u),
            "v": int(v),
            "char_start_in_edge_text": int(start),
            "char_end_in_edge_text": int(end),
        })

    parts.append("]")
    edge_text = "".join(parts)

    return edge_text, spans


def build_atom_pair_edge_text_with_spans(
    edges: List[Tuple[int, int]],
    atom_symbols: List[str],
) -> Tuple[str, List[Dict]]:

    def sym(n: int) -> str:
        return atom_symbols[n] if 0 <= n < len(atom_symbols) else "*"
    parts = ["["]
    spans: List[Dict] = []
    for i, (u, v) in enumerate(edges):
        if i > 0:
            parts.append(", ")
        start = sum(len(p) for p in parts)
        edge_str = f"({sym(u)}, {sym(v)})"
        parts.append(edge_str)
        spans.append({
            "edge_id": int(i), "u": int(u), "v": int(v),
            "char_start_in_edge_text": int(start),
            "char_end_in_edge_text": int(start + len(edge_str)),
        })
    parts.append("]")
    return "".join(parts), spans


def build_prompt(
    edge_text: str,
    prompt_mode: str = "pure_graph",
) -> Tuple[str, int]:

    td_prefix = ""

    if prompt_mode == "pure_graph":
        prefix = "[Edge List]\n"
        return td_prefix + prefix + edge_text, len(td_prefix) + len(prefix)

    if prompt_mode in ("atom_pair", "word_pair"):
        prefix = "[Edge List]\n"
        return td_prefix + prefix + edge_text, len(td_prefix) + len(prefix)

    raise ValueError(
        f"Unsupported prompt_mode={prompt_mode!r}; expected "
        "pure_graph, atom_pair, or word_pair"
    )


def shift_edge_spans_to_prompt(
    edge_spans: List[Dict],
    edge_text_start_char: int,
) -> List[Dict]:

    new_spans = []

    for sp in edge_spans:
        new_sp = dict(sp)
        new_sp["char_start"] = int(edge_text_start_char + sp["char_start_in_edge_text"])
        new_sp["char_end"] = int(edge_text_start_char + sp["char_end_in_edge_text"])
        new_spans.append(new_sp)

    return new_spans


class GraphSerializer:
    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_seq_len: int = 2048,
        prompt_mode: str = "pure_graph",
        pad_token_id: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.prompt_mode = prompt_mode
        self.pad_token_id = pad_token_id
        self._truncated_edges_total = 0

    def set_tokenizer(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @staticmethod
    def _node_batch(batch: Data) -> Tensor:
        if hasattr(batch, "batch") and batch.batch is not None:
            return batch.batch
        return torch.zeros(
            batch.x.size(0),
            dtype=torch.long,
            device=batch.x.device,
        )

    @staticmethod
    def _num_graphs(node_batch: Tensor) -> int:
        if node_batch.numel() == 0:
            return 1
        return int(node_batch.max().item()) + 1

    @staticmethod
    def _atom_symbols_for(batch: Data, local_to_orig: Tensor) -> List[str]:
        x = getattr(batch, "x", None)
        if x is None or x.dim() != 2:
            raise ValueError("atom_pair requires a two-dimensional node feature matrix")

        selected = x[local_to_orig.to(x.device)].detach().cpu()
        mutag_symbols = ("C", "N", "O", "F", "I", "Cl", "Br")
        valid_shape = selected.size(1) == len(mutag_symbols)
        near_binary = bool(
            valid_shape and torch.all((selected == 0) | (selected == 1)).item()
        )
        one_hot = bool(valid_shape and torch.all(selected.sum(dim=1) == 1).item())
        if not (valid_shape and near_binary and one_hot):
            raise ValueError(
                "atom_pair expects MUTAG seven-way one-hot node labels in the "
                "order C, N, O, F, I, Cl, Br"
            )
        return [
            mutag_symbols[int(index)]
            for index in selected.argmax(dim=1).tolist()
        ]


    def _words_for_graph(self, batch: Data, graph_id: int, local_to_orig: Tensor) -> List[str]:
        n = int(local_to_orig.numel())
        st = getattr(batch, "sentence_tokens", None)
        words: Optional[List[str]] = None
        try:
            cand = st
            if isinstance(cand, (list, tuple)) and len(cand) > graph_id and not isinstance(cand[0], str):
                cand = cand[graph_id]
            if isinstance(cand, (list, tuple)) and len(cand) == 1 and isinstance(cand[0], (list, tuple)):
                cand = cand[0]
            if isinstance(cand, str):
                cand = cand.split()
            if isinstance(cand, (list, tuple)) and (len(cand) == 0 or isinstance(cand[0], str)):
                words = [str(w) for w in cand]
        except Exception:
            words = None
        if not words:
            return [f"n{l}" for l in range(n)]
        return [words[l] if l < len(words) else f"n{l}" for l in range(n)]

    def _serialize_one_graph_in_edge_order(
        self,
        batch: Data,
        node_batch: Tensor,
        graph_id: int,
        edge_indices: Tensor,
        compact_remap: bool = False,
    ) -> SerializedGraph:
        
        device = batch.edge_index.device
        selected_edge_index = batch.edge_index[:, edge_indices]

        if compact_remap:
            if selected_edge_index.numel() > 0:
                used_nodes = torch.unique(selected_edge_index)
            else:
                used_nodes = torch.zeros(0, dtype=torch.long, device=device)
            num_nodes_kept = int(used_nodes.numel())
            old_to_new = torch.full(
                (batch.x.size(0),), -1, dtype=torch.long, device=device,
            )
            if num_nodes_kept > 0:
                old_to_new[used_nodes] = torch.arange(
                    num_nodes_kept, dtype=torch.long, device=device,
                )
        else:
            node_mask = node_batch == graph_id
            node_idx = torch.nonzero(node_mask, as_tuple=True)[0]
            num_nodes_kept = int(node_idx.numel())
            old_to_new = torch.full(
                (batch.x.size(0),), -1, dtype=torch.long, device=device,
            )
            old_to_new[node_idx] = torch.arange(
                num_nodes_kept, dtype=torch.long, device=device,
            )

        if selected_edge_index.numel() > 0:
            local_edge_index = old_to_new[selected_edge_index]
        else:
            local_edge_index = selected_edge_index

        _orig = torch.nonzero(old_to_new >= 0, as_tuple=True)[0]
        local_to_orig = torch.empty(num_nodes_kept, dtype=torch.long, device=device)
        if num_nodes_kept > 0:
            local_to_orig[old_to_new[_orig]] = _orig

        edges: List[Tuple[int, int]] = []
        local_edge_index_cpu = local_edge_index.detach().cpu()
        for eid in range(local_edge_index_cpu.size(1)):
            u = int(local_edge_index_cpu[0, eid].item())
            v = int(local_edge_index_cpu[1, eid].item())
            edges.append((u, v))

        if self.prompt_mode == "atom_pair":
            atom_symbols = self._atom_symbols_for(batch, local_to_orig)
            edge_text, edge_spans = build_atom_pair_edge_text_with_spans(
                edges, atom_symbols
            )
        elif self.prompt_mode == "word_pair":
            words = self._words_for_graph(batch, graph_id, local_to_orig)
            edge_text, edge_spans = build_atom_pair_edge_text_with_spans(
                edges, words
            )
        elif self.prompt_mode == "pure_graph":
            edge_text, edge_spans = build_edge_text_with_spans(edges)
        else:
            raise ValueError(
                f"Unsupported prompt_mode={self.prompt_mode!r}; expected "
                "pure_graph, atom_pair, or word_pair"
            )
        prompt, edge_text_start = build_prompt(
            edge_text=edge_text,
            prompt_mode=self.prompt_mode,
        )
        edge_spans = shift_edge_spans_to_prompt(
            edge_spans=edge_spans,
            edge_text_start_char=edge_text_start,
        )

        return SerializedGraph(
            prompt=prompt,
            edge_text=edge_text,
            edge_spans=edge_spans,
            num_nodes=num_nodes_kept,
            num_edges=len(edge_spans),
            graph_id=int(graph_id),
        )

    def serialize_batch(
        self,
        batch: Data,
        filter_mask: Optional[Tensor] = None,
    ) -> Tuple[List[SerializedGraph], List[Tensor]]:
        node_batch = self._node_batch(batch)
        num_graphs = self._num_graphs(node_batch)

        row = batch.edge_index[0]
        edge_batch = node_batch[row] if row.numel() > 0 else row

        compact = filter_mask is not None
        mask_dev = (
            filter_mask.to(device=row.device, dtype=torch.bool)
            if filter_mask is not None else None
        )

        records: List[SerializedGraph] = []
        edge_indices: List[Tensor] = []

        for gid in range(num_graphs):
            idx = torch.nonzero(edge_batch == gid, as_tuple=True)[0]
            if mask_dev is not None and idx.numel() > 0:
                idx = idx[mask_dev[idx]]
            rec = self._serialize_one_graph_in_edge_order(
                batch=batch,
                node_batch=node_batch,
                graph_id=gid,
                edge_indices=idx,
                compact_remap=compact,
            )
            records.append(rec)
            edge_indices.append(idx)

        return records, edge_indices

    def _tokenize_one(self, record: SerializedGraph) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        encoded = self.tokenizer(
            record.prompt,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_len,
        )

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
        offsets = encoded["offset_mapping"]

        seq_len = int(input_ids.numel())
        num_edges = int(record.num_edges)
        edge_token_matrix = torch.zeros(num_edges, seq_len, dtype=torch.float32)

        for edge_idx, span in enumerate(record.edge_spans):
            char_start = int(span["char_start"])
            char_end = int(span["char_end"])

            for token_idx, offset in enumerate(offsets):
                tok_start, tok_end = int(offset[0]), int(offset[1])
                if tok_start == 0 and tok_end == 0:
                    continue
                if tok_end > char_start and tok_start < char_end:
                    edge_token_matrix[edge_idx, token_idx] = 1.0

        graph_token_mask = (edge_token_matrix.sum(dim=0) > 0).float()
        if graph_token_mask.sum().item() == 0:
            graph_token_mask = attention_mask.float()

        return input_ids, attention_mask, edge_token_matrix, graph_token_mask

    def _pad_tokenized(
        self,
        tokenized_rows: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size = len(tokenized_rows)
        max_seq = max(int(row[0].numel()) for row in tokenized_rows)
        max_edges = max(int(row[2].size(0)) for row in tokenized_rows)

        pad_token_id = self.pad_token_id
        if pad_token_id is None and self.tokenizer is not None:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None and self.tokenizer is not None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        input_ids = torch.full(
            (batch_size, max_seq),
            fill_value=int(pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros(batch_size, max_seq, dtype=torch.long)
        edge_token_matrix = torch.zeros(batch_size, max_edges, max_seq, dtype=torch.float32)
        graph_token_mask = torch.zeros(batch_size, max_seq, dtype=torch.float32)

        for i, (ids, mask, edge_mat, graph_mask) in enumerate(tokenized_rows):
            seq_len = ids.numel()
            num_edges = edge_mat.size(0)
            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = mask
            graph_token_mask[i, :seq_len] = graph_mask
            if num_edges > 0:
                edge_token_matrix[i, :num_edges, :seq_len] = edge_mat

        return input_ids, attention_mask, edge_token_matrix, graph_token_mask

    @staticmethod
    def _pack_edge_scores(
        edge_score: Optional[Tensor],
        edge_indices: List[Tensor],
        max_edges: int,
    ) -> Optional[Tensor]:
        if edge_score is None:
            return None

        rows = []
        for idx in edge_indices:
            idx = idx.to(edge_score.device)
            score = edge_score[idx]
            if score.numel() < max_edges:
                pad = edge_score.new_zeros(max_edges - score.numel())
                score = torch.cat([score, pad], dim=0)
            rows.append(score)

        if len(rows) == 0:
            return edge_score.new_zeros(0, max_edges)

        return torch.stack(rows, dim=0)

    @staticmethod
    def _build_edge_score_mask(
        edge_indices: List[Tensor],
        max_edges: int,
        device: torch.device,
    ) -> Tensor:
        mask = torch.zeros(len(edge_indices), max_edges, dtype=torch.float32, device=device)
        for i, idx in enumerate(edge_indices):
            if idx.numel() > 0:
                mask[i, : idx.numel()] = 1.0
        return mask

    def __call__(
        self,
        batch: Data,
        subgraph_out: Optional[Any] = None,
        edge_score: Optional[Tensor] = None,
        filter_mask: Optional[Tensor] = None,
    ) -> GraphSerializerOutput:
        records, edge_indices = self.serialize_batch(batch, filter_mask=filter_mask)
        tokenized_rows = [self._tokenize_one(record) for record in records]
        input_ids, attention_mask, edge_token_matrix, graph_token_mask = self._pad_tokenized(tokenized_rows)

        if edge_score is None and subgraph_out is not None:
            edge_score = getattr(subgraph_out, "pred_edge_weight", None)

        max_edges = int(edge_token_matrix.size(1))
        packed_edge_score = self._pack_edge_scores(
            edge_score=edge_score,
            edge_indices=edge_indices,
            max_edges=max_edges,
        )

        mask_device = edge_score.device if edge_score is not None else batch.edge_index.device
        edge_score_mask = self._build_edge_score_mask(
            edge_indices=edge_indices,
            max_edges=max_edges,
            device=mask_device,
        )

        return GraphSerializerOutput(
            prompts=[record.prompt for record in records],
            records=records,
            input_ids=input_ids,
            attention_mask=attention_mask,
            edge_token_matrix=edge_token_matrix,
            graph_token_mask=graph_token_mask,
            edge_score=packed_edge_score,
            edge_score_mask=edge_score_mask,
            edge_indices=edge_indices,
        )
