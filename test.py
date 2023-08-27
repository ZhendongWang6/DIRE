from utils.config import cfg  # isort: split

import csv
import os

import torch

from utils.eval import get_val_cfg, validate
from utils.utils import get_network

cfg = get_val_cfg(cfg, split="test", copy=False)

assert cfg.ckpt_path, "Please specify the path to the model checkpoint"
model_name = os.path.basename(cfg.ckpt_path).replace(".pth", "")
dataset_root = cfg.dataset_root  # keep it
rows = []
print(f"'{cfg.exp_name}:{model_name}' model testing on...")

for i, dataset in enumerate(cfg.datasets_test):
    cfg.dataset_root = os.path.join(dataset_root, dataset)
    cfg.datasets = [""]
    model = get_network(cfg.arch)
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.cuda()
    model.eval()

    test_results = validate(model, cfg)
    print(f"{dataset}:")
    for k, v in test_results.items():
        print(f"{k}: {v:.5f}")
    print("*" * 50)
    if i == 0:
        rows.append(["TestSet"] + list(test_results.keys()))
    rows.append([dataset] + list(test_results.values()))

results_dir = os.path.join(cfg.root_dir, "data", "results")
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, f"{cfg.exp_name}-{model_name}.csv"), "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerows(rows)
