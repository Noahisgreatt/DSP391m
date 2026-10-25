"""Models 3 and 4: AI-VAD and our lighter AI-VAD Lite.

AI-VAD (Reiss & Hoshen, TMLR 2025) does not learn a network. It describes every object in a
frame by interpretable ATTRIBUTES, measures how unusual each attribute is compared with all
objects seen in the normal training videos, and adds the attributes up:

    objects   (who/what is there)      Mask R-CNN boxes + moving blobs from frame differences
    velocity  (how it moves)           optical flow inside the box -> 8 direction bins   -> GMM
    pose      (body posture)           17 body keypoints, relative to the box            -> 1-NN
    appearance (what it looks like)    CLIP image embedding of the box                   -> 1-NN

Frame score = sum over attributes of the most unusual object, calibrated on the training videos,
then smoothed over time. Each object's score says WHY it is unusual (too fast, odd pose, ...).

    model 3  aivad       the original networks (~333M parameters), code unchanged where possible
    model 4  aivad_lite  same attributes, lighter networks: YOLO11x-pose (boxes + pose in one
                         network) and RAFT optical flow (~64M), plus track-based improvements

Files:  networks.py  the networks of both models (one class per job)
        features.py  frames -> per-object attributes, cached per video
        density.py   fit on training attributes, score test attributes (AI-VAD's procedure)
        released.py  load the authors' released attributes (exact reproduction of their numbers)
        weights.py   download the pretrained network weights
"""
