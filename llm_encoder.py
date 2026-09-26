# -*- coding: utf-8 -*-
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class TokenizedGraph:
    input_ids: Tensor            # [S]
    attention_mask: Tensor       # [S]
    edge_token_matrix: Tensor    # [E, S]


@dataclass
class LLMEncoderOutput:
    embedding: Tensor                   # [B, D]
    last_hidden_state: Tensor           # [B, S, D]
    hidden_states: Optional[Tuple[Tensor, ...]]
    token_score: Optional[Tensor]       # [B, S]
    input_ids: Tensor                   # [B, S]
    attention_mask: Tensor              # [B, S]


def get_torch_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def resolve_hidden_state_index(layer: int, num_hidden_layers: int) -> int:
    if layer == -1:
        return num_hidden_layers
    if layer < -1:
        return num_hidden_layers + 1 + layer
    return layer


def last_valid_token_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    lengths = (attention_mask.long().sum(dim=1).clamp_min(1) - 1).to(hidden.device)
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, lengths]


def edge_score_to_token_score(
    edge_score: Tensor,
    edge_token_matrix: Tensor,
    attention_mask: Optional[Tensor] = None,
    score_activation: str = "sigmoid",
    per_token_norm: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    if edge_score.dim() == 1:
        edge_score = edge_score.unsqueeze(0)
    if edge_token_matrix.dim() == 2:
        edge_token_matrix = edge_token_matrix.unsqueeze(0)

    edge_score = edge_score.float()
    edge_token_matrix = edge_token_matrix.to(
        device=edge_score.device,
        dtype=edge_score.dtype,
    )
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=edge_score.device)

    if score_activation == "sigmoid":
        edge_weight = torch.sigmoid(edge_score)
    elif score_activation == "relu":
        edge_weight = torch.relu(edge_score)
    elif score_activation == "none":
        edge_weight = edge_score
    else:
        raise ValueError(f"Unknown score_activation: {score_activation}")

    if per_token_norm:
        coverage = edge_token_matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
        edge_token_matrix = edge_token_matrix / coverage

    # [B, 1, E] x [B, E, S] -> [B, S]
    token_score = torch.bmm(edge_weight.unsqueeze(1), edge_token_matrix).squeeze(1)

    if attention_mask is not None:
        token_score = token_score * attention_mask.to(token_score.device).float()

    denom = token_score.max(dim=-1, keepdim=True).values.clamp_min(eps)
    token_score = token_score / denom

    if attention_mask is not None:
        token_score = token_score * attention_mask.to(token_score.device).float()

    return token_score


class CausalTokenAttentionModifier:
    def __init__(
        self,
        gamma: float = 1.0,
        heads: Optional[Dict[int, List[int]]] = None,
        eps: float = 1e-6,
        mode: str = "pre_softmax_bias",
    ):
        self.gamma = float(gamma)
        self.heads = heads
        self.eps = float(eps)
        self.mode = mode

        self.token_score: Optional[Tensor] = None
        self.attention_mask: Optional[Tensor] = None


        self._capture_enabled: bool = False
        self._capture_layers: Optional[List[int]] = None
        self.captured: Dict[int, Dict[str, Tensor]] = {}

        self._entropy_capture_enabled: bool = False
        self._entropy_layers: Optional[List[int]] = None
        self.entropy_stats: Dict[int, Dict[str, float]] = {}

    def set_context(
        self,
        token_score: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ) -> None:
        self.token_score = token_score
        self.attention_mask = attention_mask

    def clear_context(self) -> None:
        self.token_score = None
        self.attention_mask = None

    def enable_capture(self, layers: Optional[List[int]] = None) -> None:
        self._capture_enabled = True
        self._capture_layers = None if layers is None else [int(x) for x in layers]
        self.captured = {}

    def disable_capture(self) -> None:
        self._capture_enabled = False
        self._capture_layers = None

    def enable_entropy_capture(self, layers: Optional[List[int]] = None) -> None:
        self._entropy_capture_enabled = True
        self._entropy_layers = None if layers is None else [int(x) for x in layers]
        self.entropy_stats = {}

    def disable_entropy_capture(self) -> None:
        self._entropy_capture_enabled = False
        self._entropy_layers = None

    def _should_capture_entropy(self, layer_idx: int) -> bool:
        if not self._entropy_capture_enabled:
            return False
        if self._entropy_layers is None:
            return True
        return int(layer_idx) in self._entropy_layers

    @staticmethod
    def _mean_entropy(probs: Tensor) -> float:
        """Mean over (B, H, Q) of -sum_K(p log p). Scalar in nats."""
        eps = 1e-12
        with torch.no_grad():
            p = probs.detach().float()
            ent = -(p * (p + eps).log()).sum(dim=-1).mean()
            return float(ent.item())

    def _should_capture(self, layer_idx: int) -> bool:
        if not self._capture_enabled:
            return False
        if self._capture_layers is None:
            return True
        return int(layer_idx) in self._capture_layers

    @staticmethod
    def _snapshot(t: Tensor) -> Tensor:
        return t.detach().to(device="cpu", dtype=torch.float32).clone()

    def _prepare_token_score(self, K: int, B: int, device, dtype) -> Tensor:
        ts = self.token_score.to(device=device, dtype=dtype)
        if ts.size(0) != B:
            raise ValueError(f"Batch mismatch: attn B={B}, token_score B={ts.size(0)}")
        if ts.size(1) < K:
            pad = torch.zeros(B, K - ts.size(1), device=ts.device, dtype=ts.dtype)
            ts = torch.cat([ts, pad], dim=1)
        elif ts.size(1) > K:
            ts = ts[:, :K]
        return ts

    def _prepare_key_mask(self, K: int, B: int, device, dtype) -> Optional[Tensor]:
        if self.attention_mask is None:
            return None
        am = self.attention_mask.to(device=device, dtype=dtype)
        if am.size(1) < K:
            pad = torch.zeros(B, K - am.size(1), device=am.device, dtype=am.dtype)
            am = torch.cat([am, pad], dim=1)
        elif am.size(1) > K:
            am = am[:, :K]
        return am

    def modify_logits(self, attn_logits: Tensor, layer_idx: int) -> Tensor:
        B, H, Q, K = attn_logits.shape
        token_score = self._prepare_token_score(K, B, attn_logits.device, attn_logits.dtype)
        key_mask = self._prepare_key_mask(K, B, attn_logits.device, attn_logits.dtype)
        bias = self.gamma * token_score  # [B, K]
        if key_mask is not None:
            bias = bias * key_mask
        out = attn_logits + bias[:, None, None, :]
        if self._should_capture(layer_idx):
            self.captured[int(layer_idx)] = {
                "before": self._snapshot(torch.softmax(attn_logits, dim=-1)),
                "after":  self._snapshot(torch.softmax(out, dim=-1)),
                "token_score": self._snapshot(token_score),
                "mode": "pre_softmax_bias",
            }
        if self._should_capture_entropy(layer_idx):
            H_b = self._mean_entropy(torch.softmax(attn_logits.detach().float(), dim=-1))
            H_a = self._mean_entropy(torch.softmax(out.detach().float(), dim=-1))
            self.entropy_stats[int(layer_idx)] = {
                "H_before": H_b, "H_after": H_a, "delta": H_a - H_b,
            }
        return out

    def modify_probs(self, attn_probs: Tensor, layer_idx: int) -> Tensor:
        if self.token_score is None:
            return attn_probs
        if not self._should_modify_layer(layer_idx):
            return attn_probs
        if self.mode == "pre_softmax_bias":
            return attn_probs

        B, H, Q, K = attn_probs.shape
        token_score = self._prepare_token_score(K, B, attn_probs.device, attn_probs.dtype)
        key_mask = self._prepare_key_mask(K, B, attn_probs.device, attn_probs.dtype)

        before = attn_probs if self._should_capture(layer_idx) else None

        if self.mode == "post_softmax_boost":
            out = self._modify_post_softmax_boost(attn_probs, token_score, key_mask, layer_idx)
        else:
            return attn_probs

        if before is not None:
            self.captured[int(layer_idx)] = {
                "before": self._snapshot(before),
                "after":  self._snapshot(out),
                "token_score": self._snapshot(token_score),
                "mode": self.mode,
            }
        if self._should_capture_entropy(layer_idx):
            H_b = self._mean_entropy(attn_probs)
            H_a = self._mean_entropy(out)
            self.entropy_stats[int(layer_idx)] = {
                "H_before": H_b, "H_after": H_a, "delta": H_a - H_b,
            }
        return out

    def _modify_post_softmax_boost(
        self,
        attn_probs: Tensor,
        token_score: Tensor,
        key_mask: Optional[Tensor],
        layer_idx: int,
    ) -> Tensor:
        key_weight = (1.0 + self.gamma * token_score).clamp_min(self.eps)
        if key_mask is not None:
            key_weight = key_weight * key_mask

        B, H, Q, K = attn_probs.shape

        heads = self.heads.get(int(layer_idx), [])
        heads = [int(h) for h in heads if 0 <= int(h) < H]
        if len(heads) == 0:
            return attn_probs
        out = attn_probs.clone()
        selected = out[:, heads, :, :] * key_weight[:, None, None, :]
        denom = selected.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        out[:, heads, :, :] = selected / denom
        return out

_GQA_ATTENTION_CLASSES = (
    "qwen2attention", "qwen3attention",
    "llamaattention",
    "mistralattention",
)

_MLA_ATTENTION_CLASSES = (
    "deepseekv2attention", "deepseekv3attention", "deepseekr1attention",
    "deepseek_v2attention", "deepseek_v3attention",
)


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _default_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _default_repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    bs, n_kv, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bs, n_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(bs, n_kv * n_rep, slen, head_dim)


def _import_attention_utils():
    apply_rope = None
    repeat_kv = None
    for mod_path in (
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen3.modeling_qwen3",
    ):
        try:
            mod = __import__(mod_path, fromlist=["apply_rotary_pos_emb", "repeat_kv"])
        except Exception:
            continue
        if apply_rope is None and hasattr(mod, "apply_rotary_pos_emb"):
            apply_rope = mod.apply_rotary_pos_emb
        if repeat_kv is None and hasattr(mod, "repeat_kv"):
            repeat_kv = mod.repeat_kv
        if apply_rope is not None and repeat_kv is not None:
            break
    return (apply_rope or _default_apply_rotary_pos_emb,
            repeat_kv or _default_repeat_kv)


def _resolve_family_apply_rope(module: nn.Module, default_apply_rope):
    forward_func = getattr(module.forward, "__func__", module.forward)
    family_apply_rope = getattr(forward_func, "__globals__", {}).get(
        "apply_rotary_pos_emb"
    )
    if callable(family_apply_rope):
        return family_apply_rope

    raise RuntimeError(
        f"Could not resolve the partial/interleaved RoPE function for "
        f"{module.__class__.__name__}."
    )


def _resolve_attn_dims(attn_module: nn.Module) -> Dict[str, int]:
    cfg = getattr(attn_module, "config", None)

    def _attr(obj, name, default=None):
        if obj is None:
            return default
        return getattr(obj, name, default)

    num_heads = (_attr(attn_module, "num_heads")
                 or _attr(cfg, "num_attention_heads"))
    num_kv_heads = (_attr(attn_module, "num_key_value_heads")
                    or _attr(cfg, "num_key_value_heads")
                    or num_heads)

    head_dim = (_attr(attn_module, "head_dim")
                or _attr(cfg, "head_dim"))
    if head_dim is None and cfg is not None:
        hidden_size = _attr(cfg, "hidden_size")
        if hidden_size is not None and num_heads:
            head_dim = hidden_size // num_heads

    n_kv_groups = (_attr(attn_module, "num_key_value_groups")
                   or (num_heads // num_kv_heads if (num_heads and num_kv_heads) else 1))

    scaling = (_attr(attn_module, "scaling")
               or (head_dim ** -0.5 if head_dim is not None else None))

    attn_dropout = (_attr(attn_module, "attention_dropout")
                    or _attr(cfg, "attention_dropout", 0.0)
                    or 0.0)

    return dict(
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        num_kv_groups=int(n_kv_groups),
        scaling=float(scaling),
        attention_dropout=float(attn_dropout),
    )


def _make_patched_forward(modifier: "CausalTokenAttentionModifier",
                          dims: Dict[str, int],
                          fallback_layer_idx: int,
                          apply_rope, repeat_kv_fn):

    num_heads     = dims["num_heads"]
    num_kv_heads  = dims["num_kv_heads"]
    head_dim      = dims["head_dim"]
    num_kv_groups = dims["num_kv_groups"]
    scaling       = dims["scaling"]
    attn_dropout  = dims["attention_dropout"]

    def patched_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, num_heads, head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, num_kv_heads, head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, num_kv_heads, head_dim
        ).transpose(1, 2)

        if hasattr(self, "q_norm"):
            query_states = self.q_norm(query_states)
        if hasattr(self, "k_norm"):
            key_states = self.k_norm(key_states)

        if position_embeddings is None:

            rotary = getattr(self, "rotary_emb", None)
            cos, sin = rotary(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
            )

        key_states   = repeat_kv_fn(key_states,   num_kv_groups)
        value_states = repeat_kv_fn(value_states, num_kv_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        this_layer_idx = int(getattr(self, "layer_idx", fallback_layer_idx))

        if modifier.mode == "pre_softmax_bias":
            attn_weights = modifier.modify_logits(
                attn_logits=attn_weights,
                layer_idx=this_layer_idx,
            )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32,
        ).to(query_states.dtype)

        if modifier.mode in ("post_softmax_boost"):
            attn_weights = modifier.modify_probs(
                attn_probs=attn_weights,
                layer_idx=this_layer_idx,
            )

        attn_weights = nn.functional.dropout(
            attn_weights, p=attn_dropout, training=self.training,
        )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    return patched_forward


def patch_attention(model: nn.Module, modifier: CausalTokenAttentionModifier) -> int:

    apply_rope, repeat_kv_fn = _import_attention_utils()

    patched_count = 0
    found_classes = set()
    rejected_mla: List[str] = []

    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        cls_low = cls_name.lower()

        if any(tag in cls_low for tag in _MLA_ATTENTION_CLASSES):
            rejected_mla.append(cls_name)
            continue
        if not any(tag in cls_low for tag in _GQA_ATTENTION_CLASSES):
            continue
        if not all(hasattr(module, p) for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
            continue

        dims = _resolve_attn_dims(module)
        family_apply_rope = _resolve_family_apply_rope(module, apply_rope)
        fwd = _make_patched_forward(
            modifier=modifier,
            dims=dims,
            fallback_layer_idx=patched_count,
            apply_rope=family_apply_rope,
            repeat_kv_fn=repeat_kv_fn,
        )
        module.forward = types.MethodType(fwd, module)
        patched_count += 1
        found_classes.add(cls_name)

    if patched_count == 0:
        msg = (
            "No GQA attention layers were patched. "
            "Supported families: Qwen2/Qwen3/Llama/Vicuna/Mistral. "
        )
        if rejected_mla:
            msg += (
                f"Detected MLA classes {sorted(set(rejected_mla))} "
                f"(DeepSeek-V2 / DeepSeek-R1). These use Multi-head Latent "
                f"Attention which is structurally different; a separate "
                f"patch_mla_attention is needed."
            )
        else:
            msg += (
                "Make sure the model was loaded with "
                "attn_implementation='eager' and is a standard decoder LLM."
            )
        raise RuntimeError(msg)

    print(f"[patch_attention] patched {patched_count} layers; classes={sorted(found_classes)}")
    return patched_count


patch_qwen2_attention = patch_attention


class LLMEncoder(nn.Module):

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        dtype: str = "bfloat16",
        device_map: Union[str, Dict] = "auto",
        attn_implementation: str = "eager",
        layer: int = -1,
        gamma: float = 1.0,
        per_token_norm: bool = True,
        modify_attention: bool = True,
        attention_mod_mode: str = "pre_softmax_bias",
        heads: Optional[Dict[int, List[int]]] = None,
    ):
        super().__init__()

        self.model_name_or_path = str(model_name_or_path)
        self.layer = int(layer)
        self.gamma = float(gamma)
        self.per_token_norm = bool(per_token_norm)
        self.modify_attention = bool(modify_attention)
        self.attention_mod_mode = str(attention_mod_mode)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=get_torch_dtype(dtype),
            device_map=device_map,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.freeze_llm()

        num_hidden_layers = int(self.model.config.num_hidden_layers)
        resolved_layers = list(range(num_hidden_layers // 2, num_hidden_layers))

        self.attn_modifier = CausalTokenAttentionModifier(
            gamma=gamma,
            layers=resolved_layers,
            heads=heads,
            mode=self.attention_mod_mode,
        )

        self.num_patched_layers = 0
        if self.modify_attention:

            self.num_patched_layers = patch_attention(self.model, self.attn_modifier)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def freeze_llm(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)

    def tokenize(
        self,
        texts: Union[str, List[str]],
        max_length: int = 1024,
        padding: bool = True,
        truncation: bool = True,
        return_offsets_mapping: bool = False,
    ) -> Dict[str, Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_offsets_mapping=return_offsets_mapping,
        )

    def build_edge_token_matrix_from_char_spans(
        self,
        text: str,
        edge_spans: List[Dict],
        max_length: int = 1024,
    ) -> TokenizedGraph:

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        offsets = encoded["offset_mapping"][0].tolist()

        seq_len = int(input_ids.numel())
        num_edges = len(edge_spans)

        edge_token_matrix = torch.zeros(num_edges, seq_len, dtype=torch.float32)

        for edge_idx, span in enumerate(edge_spans):
            char_start = int(span["char_start"])
            char_end = int(span["char_end"])

            for token_idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == 0 and tok_end == 0:
                    continue

                has_overlap = tok_end > char_start and tok_start < char_end
                if has_overlap:
                    edge_token_matrix[edge_idx, token_idx] = 1.0

        return TokenizedGraph(
            input_ids=input_ids,
            attention_mask=attention_mask,
            edge_token_matrix=edge_token_matrix,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        edge_token_matrix: Optional[Tensor] = None,
        edge_score: Optional[Tensor] = None,
        token_score: Optional[Tensor] = None,
        output_hidden_states: bool = True,
    ) -> LLMEncoderOutput:

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if token_score is None and edge_score is not None and edge_token_matrix is not None:
            edge_score = edge_score.to(self.device)
            edge_token_matrix = edge_token_matrix.to(self.device)

            token_score = edge_score_to_token_score(
                edge_score=edge_score,
                edge_token_matrix=edge_token_matrix,
                attention_mask=attention_mask,
                per_token_norm=self.per_token_norm,
            )

        if token_score is not None:
            token_score = token_score.to(self.device)

        if self.modify_attention:
            self.attn_modifier.set_context(
                token_score=token_score,
                attention_mask=attention_mask,
            )

        base_model = getattr(self.model, "model", self.model)
        try:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                use_cache=False,
            )
        finally:
            if self.modify_attention:
                self.attn_modifier.clear_context()

        if output_hidden_states:
            hs_idx = resolve_hidden_state_index(
                self.layer,
                self.model.config.num_hidden_layers,
            )
            hidden = outputs.hidden_states[hs_idx]
            hidden_states = outputs.hidden_states
        else:
            hidden = outputs.last_hidden_state
            hidden_states = None
        embedding = last_valid_token_pool(hidden, attention_mask)


        return LLMEncoderOutput(
            embedding=embedding,
            last_hidden_state=hidden,
            hidden_states=hidden_states,
            token_score=token_score,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )