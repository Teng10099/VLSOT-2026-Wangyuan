"""Legacy complete-label training entrypoint.

For official TRAIN/json,img data and its partial 3D labels use train_official.py.
This entrypoint no longer silently loads a hard-coded checkpoint and evaluates.
"""
import argparse
import logging
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/mono3dvg.yaml"))
    parser.add_argument("-f", "--form", choices=("1", "2", "3"), default="3")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint", required=True, help="Complete task checkpoint to initialize")
    parser.add_argument("-e", "--evaluate_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    import yaml
    from lib.helpers.model_helper import build_model
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.optimizer_helper import build_optimizer
    from lib.helpers.scheduler_helper import build_lr_scheduler
    from lib.helpers.trainer_helper import Trainer, load_initial_checkpoint
    from lib.helpers.utils_helper import set_random_seed

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_random_seed(cfg.get("random_seed", 444))
    train_loader, val_loader, _ = build_dataloader(cfg["dataset"], args.form, include_train=not args.evaluate_only)
    model, criterion = build_model(cfg["model"])
    load_initial_checkpoint(model, args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, criterion = model.to(device), criterion.to(device)
    optimizer = build_optimizer(cfg["optimizer"], model)
    scheduler, warmup = build_lr_scheduler(cfg["lr_scheduler"], optimizer, last_epoch=-1)
    logging.basicConfig(level=logging.INFO)
    train_cfg = dict(cfg["trainer"], pretrain_model=False, detr_model=False)
    trainer = Trainer(train_cfg, model, optimizer, train_loader, val_loader, scheduler, warmup,
                      logging.getLogger(__name__), criterion, cfg["model_name"])
    if args.evaluate_only:
        print(trainer.val_one_epoch(trainer.epoch))
    else:
        trainer.train()


if __name__ == "__main__":
    main()
