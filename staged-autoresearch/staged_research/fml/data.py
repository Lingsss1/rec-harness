"""Download official datasets and materialize disjoint FML splits before search.

Raw downloads are a private preparation cache. Evaluation has download=False by
construction: it receives only prepared arrays, not the complete raw archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ..storage import ProtocolError, digest, inventory, read, write
from .specs import COUNTS, DATA_PROTOCOL, DEFAULTS


def seed_hash(*values):
    return int(hashlib.md5(str(values).encode()).hexdigest(), 16) % (2 ** 31)


def carve_cifar(task, train_x, train_y, test_x=None, test_y=None):
    import numpy as np
    if task == "opacus":
        order = np.random.RandomState(42).permutation(len(train_x))
        boundary = int(0.21 * len(train_x))
        hidden, visible = order[:boundary], order[boundary:]
        x, y = train_x[visible], train_y[visible]
        ids = visible
        hidden_x, hidden_y, hidden_ids = train_x[hidden], train_y[hidden], hidden
        ratio = 0.114
    elif task == "openood":
        order = np.random.RandomState(42).permutation(len(test_x))
        boundary = int(0.3 * len(test_x))
        visible, hidden = order[:boundary], order[boundary:]
        x = np.concatenate([train_x, test_x[visible]])
        y = np.concatenate([train_y, test_y[visible]])
        ids = np.concatenate([np.arange(len(train_x)), len(train_x) + visible])
        hidden_x, hidden_y, hidden_ids = test_x[hidden], test_y[hidden], len(train_x) + hidden
        ratio = 0.057
    else:
        raise ProtocolError("Unknown CIFAR task")
    order = np.random.RandomState(42).permutation(len(x))
    boundary = int(ratio * len(x))
    val, train = order[:boundary], order[boundary:]
    return {"train": (x[train], y[train], ids[train]), "val": (x[val], y[val], ids[val]),
            "test": (hidden_x, hidden_y, hidden_ids)}


def colored_arrays(images, labels):
    """Same torch ordering, color/label noise, and env split as pinned DomainBed."""
    import numpy as np
    import torch
    torch.manual_seed(0)
    images, labels = torch.as_tensor(images), torch.as_tensor(labels)
    order = torch.randperm(len(images))
    images, labels = images[order], labels[order]
    output = {}
    for env, probability in enumerate((0.1, 0.2, 0.9)):
        x = images[env::3].clone()
        y = (labels[env::3] < 5).float()
        y = (y - (torch.rand(len(y)) < 0.25).float()).abs()
        colors = (y - (torch.rand(len(y)) < probability).float()).abs()
        x = torch.stack([x, x], dim=1)
        x[torch.arange(len(x)), (1 - colors).long(), :, :] *= 0
        x, y = x.float().div_(255), y.long()
        keys = np.arange(len(x))
        np.random.RandomState(seed_hash(0, env)).shuffle(keys)
        boundary = int(len(x) * 0.3)
        out, inside = keys[:boundary], keys[boundary:]
        # Upstream calls split_dataset(in_, 0, seed_hash(0, 2)) again for env2.
        # It changes hidden-test order, not membership.
        if env == 2:
            hidden_order = np.arange(len(inside))
            np.random.RandomState(seed_hash(0, env)).shuffle(hidden_order)
            inside = inside[hidden_order]
        ids = order[env::3].numpy()
        if env < 2:
            output["train_env" + str(env)] = (x[inside].numpy(), y[inside].numpy(), ids[inside])
        else:
            output["val"] = (x[out].numpy(), y[out].numpy(), ids[out])
            output["test"] = (x[inside].numpy(), y[inside].numpy(), ids[inside])
    output["rng_state"] = torch.get_rng_state().numpy()
    return output


def save_arrays(directory, name, arrays):
    import numpy as np
    for suffix, array in zip(("x", "y", "ids"), arrays):
        np.save(directory / (name + "_" + suffix + ".npy"), array, allow_pickle=False)


def manifest(directory, task, phase, profile, counts):
    value = {"protocol": DATA_PROTOCOL, "task": task, "phase": phase, "profile": profile,
             "counts": counts, "settings": DEFAULTS[task], "files": inventory(directory)}
    value["id"] = digest(value)
    write(directory / "data.json", value, exclusive=True)


def verify_data(directory, task, phase, allow_smoke=False):
    directory = Path(directory)
    value = read(directory / "data.json")
    if value.get("protocol") != DATA_PROTOCOL or value.get("task") != task or value.get("phase") != phase:
        raise ProtocolError("Prepared dataset task/phase/protocol mismatch")
    if value.get("profile") != "full" and not allow_smoke:
        raise ProtocolError("Smoke/fixture data cannot be used as a full FML task")
    if digest({k: v for k, v in value.items() if k != "id"}) != value["id"]:
        raise ProtocolError("Prepared dataset manifest changed")
    actual = inventory(directory)
    actual.pop("data.json")
    if actual != value["files"]:
        raise ProtocolError("Prepared dataset contents changed")
    expected = COUNTS[task]
    keys = ({"train_env0", "train_env1", "val"} if task == "colored_mnist" else {"train", "val"} | ({"ood_val"} if task == "openood" else set())) if phase == "validation" else {"test"} | ({"ood_test"} if task == "openood" else set())
    if set(value["counts"]) != keys:
        raise ProtocolError("Prepared dataset manifest has missing/unexpected partitions")
    if value["profile"] == "full" and any(expected.get(k) != v for k, v in value["counts"].items()):
        raise ProtocolError("Prepared full dataset has unexpected sample counts")
    return value


def prepare_task(task, raw, output, profile="full"):
    import numpy as np
    import torchvision.datasets as datasets
    output, raw = Path(output), Path(raw)
    if output.exists():
        for phase in ("validation", "test"):
            existing = verify_data(output / phase, task, phase, allow_smoke=profile != "full")
            if existing["profile"] != profile:
                raise ProtocolError("Use a new assets directory for a different data profile")
        return
    partial = output.with_name(output.name + ".preparing")
    owner = {"task": task, "protocol": DATA_PROTOCOL, "output": str(output.resolve()), "profile": profile}
    if partial.exists():
        marker = partial / "preparation-owner.json"
        if (marker.exists() and read(marker) == owner) or not any(partial.iterdir()):
            shutil.rmtree(partial)  # only our uncommitted preparation; raw cache remains
        else:
            raise ProtocolError("Unowned/incompatible partial data found; inspect " + str(partial))
    val, test = partial / "validation", partial / "test"
    val.mkdir(parents=True)
    test.mkdir()
    write(partial / "preparation-owner.json", owner, exclusive=True)
    if task == "colored_mnist":
        import torch
        train = datasets.MNIST(raw, train=True, download=True)
        held = datasets.MNIST(raw, train=False, download=True)
        images = torch.cat([train.data, held.data])
        labels = torch.cat([train.targets, held.targets])
        if profile != "full":
            images, labels = images[:240], labels[:240]
        arrays = colored_arrays(images, labels)
        for name in ("train_env0", "train_env1", "val"):
            save_arrays(val, name, arrays[name])
        save_arrays(test, "test", arrays["test"])
        np.save(val / "rng_state.npy", arrays["rng_state"], allow_pickle=False)
        counts = {k: len(arrays[k][0]) for k in ("train_env0", "train_env1", "val", "test")}
    else:
        train = datasets.CIFAR10(raw, train=True, download=True)
        x, y = np.asarray(train.data), np.asarray(train.targets)
        test_x = test_y = None
        if task == "openood":
            held = datasets.CIFAR10(raw, train=False, download=True)
            test_x, test_y = np.asarray(held.data), np.asarray(held.targets)
        if profile != "full":
            x, y = x[:240], y[:240]
            if test_x is not None:
                test_x, test_y = test_x[:100], test_y[:100]
        arrays = carve_cifar(task, x, y, test_x, test_y)
        for name in ("train", "val"):
            save_arrays(val, name, arrays[name])
        save_arrays(test, "test", arrays["test"])
        counts = {k: len(v[0]) for k, v in arrays.items()}
        if task == "openood":
            other = datasets.CIFAR100(raw, train=False, download=True)
            ood_x = np.asarray(other.data)
            if profile != "full":
                ood_x = ood_x[:100]
            save_arrays(val, "ood", (ood_x, np.asarray(other.targets)[:len(ood_x)], np.arange(len(ood_x))))
            # Preserve torchvision's official SVHN URL and mandatory MD5 check.
            # The upstream HTTPS hostname has a certificate mismatch; do not
            # disable TLS verification or silently accept an unverified mirror.
            other = datasets.SVHN(raw / "svhn", split="test", download=True)
            selected = np.random.RandomState(42).permutation(len(other))[:5000 if profile == "full" else 100]
            save_arrays(test, "ood", (np.transpose(other.data[selected], (0, 2, 3, 1)), other.labels[selected], selected))
            counts.update(ood_val=len(ood_x), ood_test=len(selected))
    if profile == "full" and counts != COUNTS[task]:
        raise ProtocolError("Official data count mismatch: " + json.dumps(counts))
    manifest(val, task, "validation", profile, {k: v for k, v in counts.items() if k not in ("test", "ood_test")})
    manifest(test, task, "test", profile, {k: v for k, v in counts.items() if k in ("test", "ood_test")})
    partial.rename(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(DEFAULTS))
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare_task(args.task, args.raw, args.output, "smoke" if args.smoke else "full")


if __name__ == "__main__":
    main()
