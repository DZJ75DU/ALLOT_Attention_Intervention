# -*- coding: utf-8 -*-
import argparse
import random
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from dataset import get_dataset, set_seed
from utils import EarlyStopping_Acc

from subgraph_generator import SubgraphGenerator
from serialization import GraphSerializer
from llm_encoder import LLMEncoder, edge_score_to_token_score
from classifier import build_classifier
from extractors import build_extractor_strategy
from objectives import build_objective


def parse_args():
    parser = argparse.ArgumentParser("ALLOT")
    parser.add_argument("--dataset_name", type=str, default="SPMotif-0.5")
    parser.add_argument("--dataset_root", type=str, default="/dataset")
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument("--model_root", type=str, default="/base_model")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--llm_max_seq_len", type=int, default=2048)
    parser.add_argument("--llm_device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--llm_layer", type=int, default=-1)
    parser.add_argument("--gnn_type", type=str, default="GCN")
    parser.add_argument("--gnn_hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--subgraph_ratio", type=float, default=0.3)
    parser.add_argument("--use_attention_modify", dest="use_attention_modify", action="store_true", default=True)
    parser.add_argument("--attention_gamma", type=float, default=5.5)
    parser.add_argument("--per_token_norm", dest="per_token_norm", action="store_true", default=True)
    parser.add_argument("--attention_mod_mode", type=str, default="pre_softmax_bias")
    parser.add_argument("--classifier_type", type=str, default="linear")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_metric", type=str, default="acc")
    parser.add_argument("--use_early_stopping", dest="use_early_stopping", action="store_true", default=True)
    parser.add_argument("--early_stop_patience", type=int, default=25)
    parser.add_argument("--early_stop_delta", type=float, default=0.0)
    parser.add_argument("--lambda_sparse", type=float, default=0.0)
    parser.add_argument("--lambda_entropy", type=float, default=0.0)
    parser.add_argument("--target_ratio", type=float, default=0.15)
    parser.add_argument("--extractor", type=str, default="gsat")
    parser.add_argument("--objective", type=str, default="erm")
    parser.add_argument("--gsat_temperature", type=float, default=1.0)
    parser.add_argument("--gsat_kl_lambda", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./outputs_train")

    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="pure_graph",
        choices=["pure_graph", "atom_pair", "word_pair"],
    )

    args = parser.parse_args()
    return args



def _build(dataset, seed: int):
    rng = random.Random(seed)

    by_class = {}
    for idx in range(len(dataset)):
        data = dataset[idx]
        y = int(data.y.view(-1)[0].item())
        by_class.setdefault(y, []).append(idx)

    selected = []
    for y, indices in sorted(by_class.items()):
        rng.shuffle(indices)
        selected.extend(indices)

    rng.shuffle(selected)
    return [dataset[i] for i in selected]


def build_datasets(args):
    dataset_splits, num_classes, num_features = get_dataset(
        args.dataset_name,
        data_dir=args.dataset_root,
        data_seed=args.seed,
    )

    if isinstance(dataset_splits, dict):
        train_dataset = dataset_splits["train"]
        val_dataset = dataset_splits["val"]
        test_dataset = dataset_splits["test"]
    else:
        train_dataset, val_dataset, test_dataset = dataset_splits

    _esd = getattr(args, "eval_subset_seed", None)
    _eval_base = args.seed if (_esd is None or int(_esd) < 0) else int(_esd)

    train_dataset = _build(
        train_dataset, args.seed, 
    )
    val_dataset = _build(
        val_dataset, _eval_base + 1, 
    )
    test_dataset = _build(
        test_dataset, _eval_base + 2, 
    )
    print(f"[data] train={len(train_dataset)} val={len(val_dataset)} "
          f"test={len(test_dataset)} eval_seed_base={_eval_base})")

    return train_dataset, val_dataset, test_dataset, int(num_classes), int(num_features)


def move_batch_to_device(batch, device):
    return batch.to(device)


def train_one_epoch(
    loader,
    subgraph_generator,
    serializer,
    llm_encoder,
    classifier,
    extractor_strategy,
    objective,
    optimizer,
    device,
    args,
):
    subgraph_generator.train()
    classifier.train()
    llm_encoder.eval()

    all_true = []
    all_pred = []

    total_loss = 0.0
    total_count = 0
    aux_stats = {"ce": 0.0, "penalty": 0.0, "aux": 0.0}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch.y.view(-1).long().to(device)

        optimizer.zero_grad(set_to_none=True)

        extr_out = extractor_strategy(subgraph_generator, batch, training=True)
        signals = extr_out.signals
        extractor_aux = extr_out.aux_loss

        edge_weight_for_attn = signals.edge_weight

        serialized = serializer(
            batch=batch,
            edge_score=edge_weight_for_attn,
        )


        input_ids = serialized.input_ids
        attention_mask = serialized.attention_mask
        graph_token_mask = getattr(serialized, "graph_token_mask", None)

        token_score_c = edge_score_to_token_score(
            edge_score=serialized.edge_score,
            edge_token_matrix=serialized.edge_token_matrix,
            attention_mask=attention_mask,
            score_activation="none",
            per_token_norm=args.per_token_norm,
        )

        pool_mask = graph_token_mask

        with (torch.no_grad() if not args.use_attention_modify else torch.enable_grad()):
            llm_out_c = llm_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_token_mask=pool_mask,
                token_score=token_score_c,
            )
        cls_c = classifier(llm_out_c.embedding.to(device), labels)

        total_extractor_aux = extractor_aux

        obj_out = objective(
            logits=cls_c.logits,
            labels=labels,
            extractor_aux=total_extractor_aux,
        )
        loss = obj_out.loss
        _obj_ce = obj_out.ce
        _obj_penalty = obj_out.penalty

        cls_out = cls_c
        aux_stats["ce"] += float(_obj_ce.detach().item())
        aux_stats["penalty"] += float(_obj_penalty.detach().item())
        aux_stats["aux"] += float(total_extractor_aux.detach().item())
        loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            _clip_params = list(subgraph_generator.parameters()) + list(classifier.parameters())
            torch.nn.utils.clip_grad_norm_(_clip_params, max_norm=args.grad_clip)

        optimizer.step()

        all_pred.extend(cls_out.pred.detach().cpu().tolist())
        all_true.extend(labels.detach().cpu().tolist())

        total_loss += float(loss.detach().cpu().item())
        total_count += 1

    n = max(total_count, 1)

    acc = accuracy_score(all_true, all_pred)
    bal_acc = balanced_accuracy_score(all_true, all_pred)
    return {
        "loss": total_loss / n,
        "acc": float(acc),
        "balanced_acc": float(bal_acc),
        "num_samples": len(all_true),
        "aux": {k: v / n for k, v in aux_stats.items()},
    }


@torch.no_grad()
def evaluate(
    loader,
    subgraph_generator,
    serializer,
    llm_encoder,
    classifier,
    extractor_strategy,
    device,
    args,
):
    subgraph_generator.eval()
    classifier.eval()
    llm_encoder.eval()

    all_true = []
    all_pred = []
    all_prob = []
    all_logits = []
    all_targets = []
    aux_true, aux_pred, aux_prob = [], [], []
    aux_logits_m, aux_targets_m = [], []

    total_loss = 0.0
    total_count = 0

    n_oom_skipped = 0

    def _run_batch(b):
        labels = b.y.view(-1).long().to(device)

        extr_out = extractor_strategy(subgraph_generator, b, training=False)
        signals = extr_out.signals

        edge_weight_for_attn = signals.edge_weight

        serialized = serializer(batch=b, edge_score=edge_weight_for_attn)
        token_score = edge_score_to_token_score(
            edge_score=serialized.edge_score,
            edge_token_matrix=serialized.edge_token_matrix,
            attention_mask=serialized.attention_mask,
            score_activation="none",
            per_token_norm=args.per_token_norm,
        )

        graph_token_mask_eval = getattr(serialized, "graph_token_mask", None)

        llm_out = llm_encoder(
            input_ids=serialized.input_ids,
            attention_mask=serialized.attention_mask,
            graph_token_mask=graph_token_mask_eval,
            token_score=token_score,
        )

        graph_emb = llm_out.embedding.to(device)
        cls_out = classifier(graph_emb, labels)

        r = {"count": 1, "loss": 0.0,
             "pred": [], "true": [], "prob": [], "logits_m": [], "targets_m": [],
             "aux_pred": [], "aux_true": [], "aux_prob": [],
             "aux_logits_m": [], "aux_targets_m": []}

        r["pred"] = cls_out.pred.detach().cpu().tolist()
        r["true"] = labels.detach().cpu().tolist()
        r["prob"] = cls_out.prob.detach().cpu().tolist()
        r["loss"] = float(cls_out.loss.detach().cpu().item())

        return r

    def _merge(r):
        nonlocal total_loss, total_count
        total_loss += r["loss"]
        total_count += r["count"]
        all_pred.extend(r["pred"]); all_true.extend(r["true"]); all_prob.extend(r["prob"])
        all_logits.extend(r["logits_m"]); all_targets.extend(r["targets_m"])
        aux_pred.extend(r["aux_pred"]); aux_true.extend(r["aux_true"]); aux_prob.extend(r["aux_prob"])
        aux_logits_m.extend(r["aux_logits_m"]); aux_targets_m.extend(r["aux_targets_m"])

    for batch in loader:
        try:
            _merge(_run_batch(move_batch_to_device(batch, device)))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            try:
                graphs = batch.to_data_list()
            except Exception:
                graphs = [batch]
            for g in graphs:
                try:
                    _merge(_run_batch(move_batch_to_device(Batch.from_data_list([g]), device)))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    n_oom_skipped += 1

    if n_oom_skipped:
        print(f"[warn][eval] {n_oom_skipped} graph(s) OOM'd even at batch=1 "
              f"(a huge OOD graph at seq_len={getattr(args, 'llm_max_seq_len', '?')}); "
              f"skipped them — metrics computed on the remaining graphs.")

        return out

    acc = accuracy_score(all_true, all_pred)
    bal_acc = balanced_accuracy_score(all_true, all_pred)

    out = {
        "loss": total_loss / max(total_count, 1),
        "acc": float(acc),
        "balanced_acc": float(bal_acc),
        "confusion": confusion_matrix(all_true, all_pred).tolist(),
        "num_samples": len(all_true),
    }
    return out


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir) / args.dataset_name / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, num_classes, num_features = build_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    subgraph_generator = SubgraphGenerator(
        input_dim=num_features,
        hidden_dim=args.gnn_hidden_dim,
        num_layers=args.gnn_layers,
        ratio=args.subgraph_ratio,
        gnn_type=args.gnn_type,
    ).to(device)


    model_path = Path(args.model_root, args.model)
    llm_encoder = LLMEncoder(
        model_name_or_path=model_path,
        dtype=args.dtype,
        device_map=args.llm_device_map,
        attn_implementation=args.attn_implementation,
        layer=args.llm_layer,
        gamma=args.attention_gamma,
        per_token_norm=args.per_token_norm,
        modify_attention=args.use_attention_modify,
        attention_mod_mode=args.attention_mod_mode,
    )
    llm_encoder.freeze_llm()

    serializer = GraphSerializer(
        tokenizer=llm_encoder.tokenizer,
        max_seq_len=args.llm_max_seq_len,
        prompt_mode=getattr(args, "prompt_mode", "pure_graph"),
    )

    classifier = build_classifier(
        classifier_type=args.classifier_type,
        input_dim=_cls_input_dim,
        num_classes=num_classes,
        use_layer_norm=True,
    ).to(device)

    extractor_strategy = build_extractor_strategy(args.extractor, args).to(device)
    objective = build_objective(args.objective, args).to(device)

    _groups = [
        {
            "params": list(subgraph_generator.parameters())
                        + list(objective.parameters()),
            "weight_decay": float(args.weight_decay),
        },
        {
            "params": list(classifier.parameters()),
        },
    ]
    optimizer = torch.optim.AdamW(_groups, lr=args.lr)
    print(
        f"[optimizer] per-group weight_decay: "
        f"shared={args.weight_decay}"
    )

    print(
        f"[setup] extractor={args.extractor}  objective={args.objective}  "
    )

    best_val_acc = -1.0
    best_state = None
    history = []

    early_stopper = None
    if args.use_early_stopping:
        if EarlyStopping_Acc is None:
            print(
                "[early_stopping] disabled: EarlyStopping_Acc could not be "
                "imported."
            )
        else:
            early_stopper = EarlyStopping_Acc(
                patience=int(args.early_stop_patience),
                delta=float(args.early_stop_delta),
                save_model=False,
                verbose=True,
            )
            print(
                f"[early_stopping] enabled  patience={args.early_stop_patience}  "
                f"delta={args.early_stop_delta}  max_epochs={args.epochs}"
            )

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(
            loader=train_loader,
            subgraph_generator=subgraph_generator,
            serializer=serializer,
            llm_encoder=llm_encoder,
            classifier=classifier,
            extractor_strategy=extractor_strategy,
            objective=objective,
            optimizer=optimizer,
            device=device,
            args=args,
        )

        val_result = evaluate(
            loader=val_loader,
            subgraph_generator=subgraph_generator,
            serializer=serializer,
            llm_encoder=llm_encoder,
            classifier=classifier,
            extractor_strategy=extractor_strategy,
            device=device,
            args=args,
        )

        row = {
            "epoch": epoch,
            "train": train_result,
            "val": val_result,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_acc={train_result['acc']:.4f}, "
            f"val_acc={val_result['acc']:.4f}, "
            f"train_loss={train_result['loss']:.4f}, "
            f"val_loss={val_result['loss']:.4f}  "
        )
        _val_score = -float(val_result.get("loss", 0.0))
        
        if _val_score > best_val_acc:
            best_val_acc = _val_score
            best_state = {
                "epoch": epoch,
                "val_acc": best_val_acc,
                "subgraph_generator": {
                    k: v.detach().cpu()
                    for k, v in subgraph_generator.state_dict().items()
                },
                "classifier": {
                    k: v.detach().cpu()
                    for k, v in classifier.state_dict().items()
                },
            }

        if early_stopper is not None:
            early_stopper(_val_score, model=None)
            if early_stopper.early_stop:
                _crit = "val_loss" 
                _best = -best_val_acc
                print(
                    f"[early_stopping] {_crit} has not improved by > "
                    f"{args.early_stop_delta} for {args.early_stop_patience} "
                    f"epochs; stopping at epoch {epoch} "
                    f"(best epoch = {best_state['epoch'] if best_state else 'n/a'}, "
                    f"best {_crit} = {_best:.4f})."
                )
                break

    if best_state is not None:
        subgraph_generator.load_state_dict(best_state["subgraph_generator"])
        classifier.load_state_dict(best_state["classifier"])
        subgraph_generator.to(device)
        classifier.to(device)

    print("[final_eval] selected state saved; evaluating train", flush=True)
    final_train = evaluate(
        train_loader,
        subgraph_generator,
        serializer,
        llm_encoder,
        classifier,
        extractor_strategy,
        device,
        args,
    )
    print("[final_eval] evaluating validation", flush=True)
    final_val = evaluate(
        val_loader,
        subgraph_generator,
        serializer,
        llm_encoder,
        classifier,
        extractor_strategy,
        device,
        args,
    )
    print("[final_eval] evaluating OOD test", flush=True)
    final_test = evaluate(
        test_loader,
        subgraph_generator,
        serializer,
        llm_encoder,
        classifier,
        extractor_strategy,
        device,
        args,
    )

    ckpt_path = output_dir / "checkpoint.pt"

    result_path = output_dir / "train_result.json"

    print("=" * 80)
    print("Training finished")
    print("=" * 80)
    _mk = {"acc": "acc", "balanced_acc": "balanced_acc"}.get(
        getattr(args, "eval_metric", "acc"), "acc")
    _lbl = {"acc": "acc", "balanced_acc": "balanced_acc"}[_mk]
    print("best_val_loss:", -best_val_acc)

    print(f"final_train_{_lbl}:", final_train.get(_mk))
    print(f"final_val_{_lbl}:", final_val.get(_mk))
    print(f"final_test_{_lbl}:", final_test.get(_mk))
    print(f"[both] test_acc={final_test.get('acc')}")
    print("checkpoint:", ckpt_path)
    print("result:", result_path)


if __name__ == "__main__":
    main()
