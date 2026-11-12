"""Candidate execution ONLY. A separate parent computes all reported metrics.

Training sees train arrays. Inference receives feature arrays, never evaluation
labels. OS-level denial of other files requires an audited sandbox/user boundary.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .specs import DEFAULTS


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_for(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    return torch.device(requested)


class CifarArrays(Dataset):
    def __init__(self, x, y=None, train=False):
        import torchvision.transforms as transforms
        self.x, self.y = x, y
        operations = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
        self.transform = transforms.Compose(operations + [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        from PIL import Image
        x = self.transform(Image.fromarray(self.x[index]))
        return (x, int(self.y[index])) if self.y is not None else x


def load_xy(data, name):
    return (np.load(data / (name + "_x.npy"), mmap_mode="r", allow_pickle=False),
            np.load(data / (name + "_y.npy"), mmap_mode="r", allow_pickle=False))


def load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pinned DomainBed torch 1.12 does not expose weights_only
        return torch.load(path, map_location=device)


def import_algorithm(source, task):
    sys.path.insert(0, str(source))
    return importlib.import_module("domainbed.algorithms" if task == "colored_mnist" else "algorithm")


def validate_config(config, task):
    json.dumps(config, allow_nan=False)
    def positive(name, integer=False):
        v = config[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 or (integer and not isinstance(v, int)):
            raise ValueError("Invalid training configuration: " + name)
    positive("batch_size", True)
    positive("epochs", True)
    positive("lr")
    v = config["momentum"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v < 1:
        raise ValueError("Invalid momentum")
    if task == "opacus":
        if config["target_epsilon"] != 8 or config["target_delta"] != 1e-5:
            raise ValueError("Fixed privacy target changed")
        positive("max_grad_norm")
    else:
        decay = config["weight_decay"]
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or decay < 0:
            raise ValueError("Invalid weight decay")
        positive("lr_gamma")
        if not isinstance(config["lr_milestones"], list) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in config["lr_milestones"]):
            raise ValueError("Invalid LR milestones")


def domainbed_train(module, source, data, result, settings, device):
    from domainbed import hparams_registry
    from domainbed.lib.fast_data_loader import InfiniteDataLoader
    hparams = hparams_registry.default_hparams("ERM", "ColoredMNIST")
    torch.set_rng_state(torch.from_numpy(np.load(data / "rng_state.npy", allow_pickle=False).copy()))
    loaders = []
    for env in (0, 1):
        x, y = load_xy(data, "train_env" + str(env))
        dataset = TensorDataset(torch.from_numpy(x.copy()), torch.from_numpy(y.copy()))
        loaders.append(InfiniteDataLoader(dataset, None, hparams["batch_size"], settings["workers"]))
    algorithm = module.get_algorithm_class("ERM")((2, 28, 28), 2, 2, hparams).to(device)
    batches = zip(*loaders)
    for step in range(settings["steps"]):
        minibatches = [(x.to(device), y.to(device)) for x, y in next(batches)]
        algorithm.update(minibatches, None)
        if step % 100 == 0 or step == settings["steps"] - 1:
            print("train step %d/%d" % (step + 1, settings["steps"]), flush=True)
    torch.save({k: v.detach().cpu() for k, v in algorithm.state_dict().items()}, result / "model.pt")
    return {"hparams": hparams, "steps": settings["steps"]}


def cifar_train(module, task, data, result, settings, device):
    config = module.get_training_config()
    validate_config(config, task)
    config = dict(config)
    if settings.get("smoke"):
        config.update(epochs=1, batch_size=16)
    x, y = load_xy(data, "train")
    dataset = CifarArrays(x, y, train=True)
    if config["batch_size"] > len(dataset):
        raise ValueError("Batch size exceeds training population")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        num_workers=settings["workers"], drop_last=task == "opacus", pin_memory=device.type == "cuda")
    model = module.get_model(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
        weight_decay=config.get("weight_decay", 0) if task == "openood" else 0)
    privacy = None
    if task == "opacus":
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        ModuleValidator.validate(model, strict=True)
        privacy = PrivacyEngine(accountant="prv")
        model, optimizer, loader = privacy.make_private_with_epsilon(module=model, optimizer=optimizer,
            data_loader=loader, epochs=config["epochs"], target_epsilon=8, target_delta=1e-5,
            max_grad_norm=config["max_grad_norm"])
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["lr_milestones"], gamma=config["lr_gamma"])
    logical_steps = config["epochs"] * len(loader)
    for epoch in range(config["epochs"]):
        model.train()
        if privacy is not None:
            from opacus.utils.batch_memory_manager import BatchMemoryManager
            context = BatchMemoryManager(data_loader=loader, max_physical_batch_size=settings["max_physical_batch_size"], optimizer=optimizer)
        else:
            import contextlib
            context = contextlib.nullcontext(loader)
        with context as batches:
            for inputs, targets in batches:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                # Opacus' empty Poisson samples are handled by its GradSampleModule.
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
        if privacy is None:
            scheduler.step()
        print("train epoch %d/%d" % (epoch + 1, config["epochs"]), flush=True)
    model = model._module if privacy is not None else model
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, result / "model.pt")
    spec = {"training_config": config, "logical_steps": logical_steps, "population": len(dataset)}
    if privacy is not None:
        spec["privacy"] = {"mechanism": "prv", "history": privacy.accountant.history,
            "sample_rate": float(loader.sample_rate), "noise_multiplier": float(optimizer.noise_multiplier)}
    return spec


def train(args):
    settings = dict(DEFAULTS[args.task])
    if args.smoke:
        settings.update(steps=2, workers=0, max_physical_batch_size=8, smoke=True)
    seed_everything(settings["seed"])
    device = device_for(args.device)
    module = import_algorithm(args.source, args.task)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if args.task == "colored_mnist":
        spec = domainbed_train(module, args.source, args.data, args.output, settings, device)
    else:
        spec = cifar_train(module, args.task, args.data, args.output, settings, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    spec.update(task=args.task, settings=settings, training_seconds=time.monotonic() - started)
    write_json(args.output / "model_spec.json", spec)


def infer(args):
    seed_everything(DEFAULTS[args.task]["seed"])
    device = device_for(args.device)
    module = import_algorithm(args.source, args.task)
    spec = json.loads((args.checkpoint / "model_spec.json").read_text())
    if spec["task"] != args.task:
        raise ValueError("Checkpoint task mismatch")
    if args.task == "colored_mnist":
        model = module.get_algorithm_class("ERM")((2, 28, 28), 2, 2, spec["hparams"]).to(device)
    else:
        model = module.get_model(num_classes=10).to(device)
    model.load_state_dict(load_state(args.checkpoint / "model.pt", device), strict=True)
    model.eval()
    x = np.load(args.inputs, allow_pickle=False, mmap_mode="r")
    dataset = TensorDataset(torch.from_numpy(x.copy())) if args.task == "colored_mnist" else CifarArrays(x)
    loader = DataLoader(dataset, batch_size=DEFAULTS[args.task]["eval_batch_size"], shuffle=False, num_workers=0)
    values = []
    for batch in loader:
        features = (batch[0] if args.task == "colored_mnist" else batch).to(device)
        if args.task == "openood":
            prediction = module.compute_ood_score(model, features)
        else:
            with torch.no_grad():
                prediction = model.predict(features) if args.task == "colored_mnist" else model(features)
        values.append(prediction.detach().cpu().numpy())
    predictions = np.concatenate(values)
    if len(predictions) != len(x) or not np.isfinite(predictions).all():
        raise ValueError("Nonfinite or incorrectly sized model predictions")
    np.save(args.output, predictions, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(DEFAULTS), required=True)
    parser.add_argument("--action", choices=("train", "infer"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("STAGED_TORCH_THREADS", "1")))
    (train if args.action == "train" else infer)(args)


if __name__ == "__main__":
    main()
