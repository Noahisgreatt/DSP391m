"""The teacher's rule: no model may come from a library call (timm, ultralytics/YOLO, CLIP,
detectron2, the hiera package). Block those libraries and build all four models: everything must
still work."""
import subprocess
import sys

CODE = r"""
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in ('timm', 'ultralytics', 'clip', 'detectron2', 'hiera'):
            raise ModuleNotFoundError('blocked ' + name)
sys.meta_path.insert(0, Block())
import torch
from src.models import astt, mulde
from src.utils import load_config
for n in ('astt_baseline', 'astt_improved'):
    astt.build(load_config(f'configs/{n}.yaml'), pretrained=False)
for n in ('mulde', 'mulde_improved'):
    mulde.Features(load_config(f'configs/{n}.yaml'), torch.device('cpu'), weights=False)
import src.data, src.behavior, src.train, src.evaluate, src.demo
print('OK')
"""


def test_runs_without_wrapper_libraries():
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]
