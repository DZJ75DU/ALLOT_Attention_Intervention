#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os.path as osp
import pickle as pkl
import random
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs, read_tu_data


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


class SPMotif(InMemoryDataset):
    splits = ("train", "val", "test")

    def __init__(
        self,
        root,
        mode="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        if mode not in self.splits:
            raise ValueError(f"Unsupported SPMotif split: {mode}")
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        index = self.processed_file_names.index(f"SPMotif_{mode}.pt")
        self.data, self.slices = safe_torch_load(self.processed_paths[index])

    @property
    def raw_file_names(self):
        return ["train.npy", "val.npy", "test.npy"]

    @property
    def processed_file_names(self):
        return ["SPMotif_train.pt", "SPMotif_val.pt", "SPMotif_test.pt"]

    def download(self):
        expected = osp.join(self.raw_dir, "train.npy")
        if not osp.exists(expected):
            raise FileNotFoundError(f"SPMotif raw file not found: {expected}")

    def process(self):
        raw_path = osp.join(self.raw_dir, f"{self.mode}.npy")
        edge_indices, labels, edge_labels, role_ids, positions = np.load(
            raw_path, allow_pickle=True
        )
        data_list = []
        for index, (edge_index, label, edge_label, role_id, pos) in enumerate(
            zip(edge_indices, labels, edge_labels, role_ids, positions)
        ):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
            nodes = torch.unique(edge_index)
            if int(nodes.max()) != nodes.numel() - 1:
                raise ValueError(f"SPMotif graph {index} has non-contiguous node ids")

            z = torch.as_tensor(role_id, dtype=torch.long)
            node_label = z.float()
            node_label[node_label != 0] = 1.0
            edge_label = torch.as_tensor(edge_label, dtype=torch.float).view(-1)
            data = Data(
                x=torch.rand((nodes.numel(), 4), dtype=torch.float),
                y=torch.tensor([int(label)], dtype=torch.long),
                z=z,
                edge_index=edge_index,
                edge_attr=torch.ones(edge_index.size(1), 1, dtype=torch.float),
                node_label=node_label,
                edge_label=edge_label,
                pos=torch.as_tensor(pos) if not torch.is_tensor(pos) else pos,
                edge_gt_att=edge_label.long(),
                name=f"SPMotif-{self.mode}-{index}",
                idx=index,
            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        output_index = self.processed_file_names.index(
            f"SPMotif_{self.mode}.pt"
        )
        torch.save(self.collate(data_list), self.processed_paths[output_index])


def split_sentigraph_by_density(dataset):
    test_indices, train_pool = [], []
    for index, graph in enumerate(dataset):
        if graph.num_edges <= 2:
            continue
        density = float(graph.num_edges) / float(graph.num_nodes)
        if density >= 1.76785714:
            train_pool.append(index)
        elif density <= 1.57142857:
            test_indices.append(index)
    val_size = int(len(train_pool) * 0.1)
    val_indices = train_pool[:val_size]
    train_indices = train_pool[val_size:]
    return (
        dataset[train_indices],
        dataset[val_indices],
        dataset[test_indices],
    )


def undirected_graph(data: Data) -> Data:
    reverse_edges = torch.stack([data.edge_index[1], data.edge_index[0]], dim=0)
    data.edge_index = torch.cat([reverse_edges, data.edge_index], dim=1)
    return data


def _split_big_sentigraph_data(data: Data, batch: np.ndarray):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()
    slices = {
        "x": node_slice,
        "edge_index": edge_slice,
        "y": torch.arange(0, batch[-1] + 2, dtype=torch.long),
        "sentence_tokens": torch.arange(0, batch[-1] + 2, dtype=torch.long),
        "name": torch.arange(0, batch[-1] + 2, dtype=torch.long),
    }
    return data, slices


def _read_sentigraph_file(folder: str, prefix: str, name: str):
    path = osp.join(folder, f"{prefix}_{name}.txt")
    return np.genfromtxt(path, dtype=np.int64)


def read_sentigraph_data(folder: str, prefix: str):
    with open(osp.join(folder, f"{prefix}_node_features.pkl"), "rb") as handle:
        x = torch.from_numpy(pkl.load(handle)).float()
    edge_index = torch.tensor(
        _read_sentigraph_file(folder, prefix, "edge_index"), dtype=torch.long
    ).T
    batch = _read_sentigraph_file(folder, prefix, "node_indicator") - 1
    y = torch.tensor(
        _read_sentigraph_file(folder, prefix, "graph_labels"), dtype=torch.long
    )

    supplement = {}
    split_path = osp.join(folder, f"{prefix}_split_indices.txt")
    if osp.exists(split_path):
        supplement["split_indices"] = torch.tensor(
            _read_sentigraph_file(folder, prefix, "split_indices"),
            dtype=torch.long,
        )

    sentence_path = osp.join(folder, f"{prefix}_sentence_tokens.json")
    if osp.exists(sentence_path):
        with open(sentence_path, encoding="utf-8") as handle:
            sentence_tokens = list(json.load(handle).values())
    else:
        sentence_tokens = [""] * int(y.numel())

    data = Data(
        name=torch.arange(y.numel()),
        x=x,
        edge_index=edge_index,
        y=y,
        sentence_tokens=sentence_tokens,
    )
    data, slices = _split_big_sentigraph_data(data, batch)
    return data, slices, supplement


class SentiGraphTransform:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, data):
        data.edge_attr = torch.ones(data.edge_index.size(1), 1)
        return self.transform(data) if self.transform is not None else data


class SentiGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super().__init__(root, transform, pre_transform)
        output = safe_torch_load(self.processed_paths[0])
        self.supplement = {}
        if isinstance(output, (list, tuple)):
            data = output[0]
            self.slices = output[1] if len(output) > 1 else None
            data_class = output[3] if len(output) >= 4 else Data
            if len(output) == 3 and isinstance(output[2], dict):
                self.supplement = output[2]
        else:
            data, self.slices, data_class = output, None, Data
        self.data = data if not isinstance(data, dict) else data_class.from_dict(data)

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return [
            f"{self.name}_node_features.pkl",
            f"{self.name}_node_indicator.txt",
            f"{self.name}_sentence_tokens.json",
            f"{self.name}_edge_index.txt",
            f"{self.name}_graph_labels.txt",
        ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        self.data, self.slices, self.supplement = read_sentigraph_data(
            self.raw_dir, self.name
        )
        if self.pre_transform is not None:
            data_list = [self.pre_transform(self.get(i)) for i in range(len(self))]
            self.data, self.slices = self.collate(data_list)
        torch.save(
            (self.data, self.slices, self.supplement), self.processed_paths[0]
        )


class TUDataset(InMemoryDataset):
    url = "https://www.chrsmrrs.com/graphkerneldatasets"
    cleaned_url = (
        "https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets"
    )

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
        cleaned: bool = False,
    ) -> None:
        self.name = name
        self.cleaned = cleaned
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )
        output = fs.torch_load(self.processed_paths[0])
        if not isinstance(output, tuple) or len(output) not in (3, 4):
            raise RuntimeError("Unsupported processed TUDataset format")
        if len(output) == 3:
            data, self.slices, self.sizes = output
            data_class = Data
        else:
            data, self.slices, self.sizes, data_class = output
        self.data = data if not isinstance(data, dict) else data_class.from_dict(data)

        if self._data.x is not None and not use_node_attr:
            self._data.x = self._data.x[:, self.num_node_attributes :]
        if self._data.edge_attr is not None and not use_edge_attr:
            self._data.edge_attr = self._data.edge_attr[:, self.num_edge_attributes :]

    @property
    def raw_dir(self) -> str:
        suffix = "_cleaned" if self.cleaned else ""
        return osp.join(self.root, self.name, f"raw{suffix}")

    @property
    def processed_dir(self) -> str:
        suffix = "_cleaned" if self.cleaned else ""
        return osp.join(self.root, self.name, f"processed{suffix}")

    @property
    def num_node_labels(self) -> int:
        return self.sizes["num_node_labels"]

    @property
    def num_node_attributes(self) -> int:
        return self.sizes["num_node_attributes"]

    @property
    def num_edge_labels(self) -> int:
        return self.sizes["num_edge_labels"]

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes["num_edge_attributes"]

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f"{self.name}_A.txt",
            f"{self.name}_graph_indicator.txt",
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        url = self.cleaned_url if self.cleaned else self.url
        fs.cp(f"{url}/{self.name}.zip", self.raw_dir, extract=True)
        nested_dir = osp.join(self.raw_dir, self.name)
        for filename in fs.ls(nested_dir):
            fs.mv(filename, osp.join(self.raw_dir, osp.basename(filename)))
        fs.rm(nested_dir)

    def process(self) -> None:
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(index) for index in range(len(self))]
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
            self._data_list = None
        fs.torch_save(
            (self._data.to_dict(), self.slices, sizes, self._data.__class__),
            self.processed_paths[0],
        )


def _dataset_info_from_splits(splits):
    first = splits[0]
    num_classes = getattr(first, "num_classes", None)
    num_features = getattr(first, "num_features", None)
    if num_classes is None:
        labels = [
            int(split[index].y.view(-1)[0].item())
            for split in splits
            for index in range(len(split))
        ]
        num_classes = len(set(labels)) if labels else 0
    if num_features is None:
        x = first[0].x
        num_features = int(x.size(-1)) if x is not None else 0
    return num_classes, num_features


def get_dataset(
    dataset_name: str,
    data_dir: str,
    data_seed: int = 42,
    sparse: bool = True,
    cleaned: bool = False,
    **_unused,
):
    del sparse
    set_seed(data_seed)
    normalized = normalize_name(dataset_name)

    if normalized in {"spmotif-0.5", "spmotif-0.7", "spmotif-0.9"}:
        root = osp.join(data_dir, dataset_name)
        splits = [SPMotif(root, mode=mode) for mode in ("train", "val", "test")]
        num_classes, num_features = _dataset_info_from_splits(splits)
        return splits, num_classes, num_features

    if normalized == "graph-sst2":
        dataset = SentiGraphDataset(
            root=data_dir,
            name="Graph-SST2",
            transform=SentiGraphTransform(),
        )
        splits = list(split_sentigraph_by_density(dataset))
        num_classes, num_features = _dataset_info_from_splits(splits)
        return splits, num_classes, num_features

    if normalized == "mutag":
        dataset = TUDataset(data_dir, "MUTAG", cleaned=cleaned)
        train_indices, val_indices, test_indices = [], [], []
        for index, data in enumerate(dataset):
            size = int(data.x.size(0)) if data.x is not None else int(data.num_nodes)
            if size <= 15:
                train_indices.append(index)
            elif size <= 20:
                val_indices.append(index)
            else:
                test_indices.append(index)
        return (
            [
                dataset[train_indices],
                dataset[val_indices],
                dataset[test_indices],
            ],
            dataset.num_classes,
            dataset.num_features,
        )