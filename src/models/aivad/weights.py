"""Download what models 3 / 4 need (run on your machine; needs internet).

    python -m src.models.aivad.weights --model aivad          # model 3 network weights
    python -m src.models.aivad.weights --model aivad_lite     # model 4 (YOLO11x-pose, RAFT)
    python -m src.models.aivad.weights --released             # the authors' attribute files

Sources (all from the AI-VAD / AlphaPose READMEs):
  Mask R-CNN R50-FPN 3x   detectron2 model zoo (model_final_f10217.pkl)
  FlowNet2                AI-VAD pre_processing/README.md (Google Drive)
  AlphaPose FastPose R50  AlphaPose docs/MODEL_ZOO.md (Google Drive)
  CLIP ViT-B/16           downloaded by the `clip` package on first use (~/.cache/clip)
  YOLO11x-pose, RAFT      downloaded by ultralytics / torchvision on first use
  released attributes     AI-VAD README "our set of representations" (Google Drive folder) +
                          the training pose file + the test pose file inside the AI-VAD repo
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from src.config import load_config

MASKRCNN_URL = ("https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/"
                "mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl")
GDRIVE = {"flownet2": "1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da",
          "fastpose": "1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn"}
RELEASED_FOLDER = "1vSMpDb5jIyc2tNJaYVphguUlFcwPayms"
TRAIN_POSE = "1fxMmmZ8TmmdGOovbC2QYPgWTaRxyVdE0"


def get(path: Path, url=None, gdrive_id=None):
    if path.exists():
        print(f"ok  {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {path.name} ...")
    if gdrive_id:
        import gdown
        gdown.download(id=gdrive_id, output=str(path), quiet=False)
    else:
        urllib.request.urlretrieve(url, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["aivad", "aivad_lite"])
    ap.add_argument("--released", action="store_true")
    args = ap.parse_args()
    if args.model == "aivad":
        w = load_config("configs/aivad.yaml")["weights"]
        get(Path(w["maskrcnn"]), url=MASKRCNN_URL)
        get(Path(w["flownet2"]), gdrive_id=GDRIVE["flownet2"])
        get(Path(w["fastpose"]), gdrive_id=GDRIVE["fastpose"])
        import clip
        clip.load("ViT-B/16", device="cpu")
        print("ok  CLIP ViT-B/16")
    if args.model == "aivad_lite":
        from ultralytics import YOLO
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        YOLO(load_config("configs/aivad_lite.yaml")["weights"]["yolo_pose"])
        raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2)
        print("ok  YOLO11x-pose + RAFT")
    if args.released:
        import gdown
        cfg = load_config("configs/aivad.yaml")
        dst = Path(cfg["model"]["released_dir"])
        tmp = dst.parent / "_drive"
        if not (dst / "test" / "velocity.npy").exists():
            gdown.download_folder(id=RELEASED_FOLDER, output=str(tmp), quiet=False)
            found = sorted(tmp.rglob("velocity.npy"))
            print("downloaded:", *[str(p) for p in found], sep="\n  ")
            for p in found:                                         # .../shanghaitech/<split>/x.npy
                if "shanghaitech" in str(p).lower():
                    split = p.parent.name
                    (dst / split).mkdir(parents=True, exist_ok=True)
                    for f in p.parent.glob("*.npy"):
                        shutil.copy2(f, dst / split / f.name)
        get(dst / "train" / "pose.npy", gdrive_id=TRAIN_POSE)
        test_pose = Path("third_party/AIVAD/extracted_features/shanghaitech/test/pose.npy")
        if not (dst / "test" / "pose.npy").exists():
            (dst / "test").mkdir(parents=True, exist_ok=True)
            shutil.copy2(test_pose, dst / "test" / "pose.npy")
        print(f"released attributes in {dst}:", sorted(str(p.relative_to(dst)) for p in dst.rglob("*.npy")))


if __name__ == "__main__":
    main()
