"""Step 1, DATA.   prepare.py: get, unpack, split (command line)   loaders.py: feed the models.

    python -m src.dataset prepare | manual | split | download | link DIR...
"""
from src.dataset.loaders import VideoWindows, train_dataset, video_dataset                  # noqa: F401
from src.dataset.prepare import (data_card, extract_video, load_split,           # noqa: F401
                                 make_split, prepare_manual, prepare_shanghaitech, video_labels)
