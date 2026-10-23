"""Loaders for upstream complete labels and separately packaged official clips."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader


def my_worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(cfg, form="3", workers=None, include_train=True):
    """Always return (train, val, test); evaluation can omit training loading."""
    form = str(form)
    if cfg.get("type") != "Mono3DRefer" or form not in ("1", "2", "3"):
        raise NotImplementedError("Unsupported dataset/form: {!r}/{!r}".format(cfg.get("type"), form))
    from lib.datasets.mono3drefer.mono3drefer_dataset import Mono3DRefer_Dataset
    workers = int(cfg.get("workers", 0) if workers is None else workers)
    sets = [
        Mono3DRefer_Dataset(split=cfg["train_split"], cfg=cfg, form=form) if include_train else None,
        Mono3DRefer_Dataset(split=cfg["val_split"], cfg=cfg, form=form),
        Mono3DRefer_Dataset(split=cfg["test_split"], cfg=cfg, form=form),
    ]
    generator = torch.Generator().manual_seed(int(cfg.get("random_seed", 444)))
    return tuple(None if dataset is None else DataLoader(
        dataset, batch_size=int(cfg.get("batch_size", 1)), num_workers=workers,
        worker_init_fn=my_worker_init_fn, shuffle=index == 0,
        pin_memory=bool(cfg.get("pin_memory", False)), drop_last=False,
        generator=generator,
    ) for index, dataset in enumerate(sets))


def build_official_dataloader(dataset, batch_size=1, workers=0, shuffle=False,
                              seed=444, pin_memory=False):
    """Default collate preserves list[frame_dict] for frame-streamed clips."""
    if int(batch_size) != 1:
        raise ValueError("The inherited TTM is configured for batch_size=1")
    if len(dataset) == 0:
        raise ValueError("Official dataset selection is empty")
    return DataLoader(
        dataset, batch_size=1, num_workers=int(workers), shuffle=bool(shuffle),
        pin_memory=bool(pin_memory), drop_last=False,
        worker_init_fn=my_worker_init_fn,
        generator=torch.Generator().manual_seed(int(seed)),
    )
