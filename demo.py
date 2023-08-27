import argparse
import glob
import os

import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from utils.utils import get_network, str2bool, to_cuda

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-f", "--file", default="data/test/lsun_adm/1_fake/0.png", type=str, help="path to image file or directory of images"
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="data/exp/ckpt/lsun_adm/model_epoch_latest.pth",
)
parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
parser.add_argument("--arch", type=str, default="resnet50")
parser.add_argument("--aug_norm", type=str2bool, default=True)

args = parser.parse_args()

if os.path.isfile(args.file):
    print(f"Testing on image '{args.file}'")
    file_list = [args.file]
elif os.path.isdir(args.file):
    file_list = sorted(glob.glob(os.path.join(args.file, "*.jpg")) + glob.glob(os.path.join(args.file, "*.png"))+glob.glob(os.path.join(args.file, "*.JPEG")))
    print(f"Testing images from '{args.file}'")
else:
    raise FileNotFoundError(f"Invalid file path: '{args.file}'")


model = get_network(args.arch)
state_dict = torch.load(args.model_path, map_location="cpu")
if "model" in state_dict:
    state_dict = state_dict["model"]
model.load_state_dict(state_dict)
model.eval()
if not args.use_cpu:
    model.cuda()

print("*" * 50)

trans = transforms.Compose(
    (
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    )
)
for img_path in tqdm(file_list, dynamic_ncols=True, disable=len(file_list) <= 1):
    img = Image.open(img_path).convert("RGB")
    img = trans(img)
    if args.aug_norm:
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    in_tens = img.unsqueeze(0)
    if not args.use_cpu:
        in_tens = in_tens.cuda()

    with torch.no_grad():
        prob = model(in_tens).sigmoid().item()
    print(f"Prob of being synthetic: {prob:.4f}")
