"""Pinned upstream contracts. No downloads or training on import."""
FML_COMMIT = "d336651ebea50c622c256f02ded82b68b4451fdc"
DOMAINBED_COMMIT = "b93c22a1cfc3b2428398272c1a116c8de1f4139e"
DATA_PROTOCOL = "staged-fml-v1"
MICROMAMBA_VERSION = "2.3.3"
MICROMAMBA_SHA256 = "e7274528ceb9c20d048a428d6c22d7e02e268f8ffb762c4c365422347c8b8ba2"
CLAUDE_VERSION = "2.1.146"
ENVIRONMENTS = {
    "colored_mnist": {"python": "3.10.16", "torch": "1.12.1", "torchvision": "0.13.1", "cuda": "cu113",
        "packages": ["numpy==1.22.4", "scipy==1.9.3", "Pillow==10.3.0", "timm==0.9.16", "tqdm==4.66.4"]},
    "openood": {"python": "3.9.21", "torch": "2.5.1", "torchvision": "0.20.1", "cuda": "cu121",
        "packages": ["numpy==2.0.2", "scipy==1.13.1", "scikit-learn==1.6.1", "Pillow==10.3.0"]},
    "opacus": {"python": "3.10.16", "torch": "2.5.1", "torchvision": "0.20.1", "cuda": "cu121",
        "packages": ["opacus==1.5.4", "numpy==2.2.6", "scipy==1.15.3", "Pillow==10.3.0"]},
}
DEFAULTS = {
    "colored_mnist": {"seed": 0, "steps": 5001, "workers": 8, "eval_batch_size": 64},
    "openood": {"seed": 42, "workers": 4, "eval_batch_size": 256},
    "opacus": {"seed": 42, "workers": 2, "eval_batch_size": 256, "max_physical_batch_size": 64},
}
COUNTS = {
    "colored_mnist": {"train_env0": 16334, "train_env1": 16334, "val": 6999, "test": 16334},
    "openood": {"train": 49979, "val": 3021, "ood_val": 10000, "test": 7000, "ood_test": 5000},
    "opacus": {"train": 34997, "val": 4503, "test": 10500},
}
