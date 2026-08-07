# Vision-Based Line Following for a Mobile Robot

A 5-class steering classifier (ResNet-18) that drives a mobile robot around a
taped track in real time, deployed on a Jetson Orin Nano with a ZED2i stereo
camera and a Yukon serial motor controller.

Full write-up: [`docs/report.pdf`](docs/report.pdf)

## Summary

The task is framed as per-frame 5-class classification
(`hard_left`, `left`, `straight`, `right`, `hard_right`) rather than steering
regression, since the robot's motor interface only exposes discrete named
commands. A ResNet-18 pretrained on ImageNet is fine-tuned on 5,632
manually-labelled frames collected across 12 laps (6 per direction), with
predictions smoothed by a confidence-gated 3-frame majority vote before being
mapped to a motor command.

**Main finding:** test accuracy was a poor predictor of deployed performance.
The highest-accuracy variant (v3, 80.5%) systematically failed at
right-turning corners, while a lower-accuracy variant (v2, 77.7%) with better
recall on the rare `right` class was the one that actually worked on the
track. Per-corner testing confirmed the failure was class-specific
(right-turn recall) rather than direction-specific.

| Model | Test acc | hard_left | left | straight | right | hard_right | Notes |
|---|---|---|---|---|---|---|---|
| v1 | 71.2% | 70 | 45 | 85 | 24 | 80 | Baseline, noisy labels |
| v2 | 77.7% | **89** | **52** | 77 | **42** | 80 | Relabelled, full inverse-frequency sampling — **used for deployment** |
| v3 | 80.5% | 78 | 48 | **86** | 33 | **84** | Relabelled, softened (w^0.3) sampling |

(per-class recall, %, on the held-out test set)

## Pipeline

```
Videos/  ──01──▶  Frames/  ──02/03──▶  labelled frames  ──04──▶  best_model_v{2,3}.pth  ──05/06──▶  deployed robot
```

| Notebook | Purpose |
|---|---|
| [`01_video_to_frames.ipynb`](notebooks/01_video_to_frames.ipynb) | Extracts every 5th frame from the raw lap recordings |
| [`02_label_frames.ipynb`](notebooks/02_label_frames.ipynb) | Manual keyboard-driven labelling tool (`h`/`l`/`s`/`r`/`k`) |
| [`03_relabelling.ipynb`](notebooks/03_relabelling.ipynb) | Targeted re-label pass that fixed "labeller bias" — frames labelled by the driver's *intended* turn rather than what was visually on screen. Corrected 269 frames, mainly `right`→`straight` |
| [`04_train_model.ipynb`](notebooks/04_train_model.ipynb) | Video-level train/val/test split, ResNet-18 fine-tuning with class-weighted loss + `WeightedRandomSampler`, evaluation and confusion matrices for v2 and v3 |
| [`05_deploy_spin_burst.ipynb`](notebooks/05_deploy_spin_burst.ipynb) | Deployment loop, control strategy S1: discrete spin-then-nudge at each turn |
| [`06_deploy_differential_drive.ipynb`](notebooks/06_deploy_differential_drive.ipynb) | Deployment loop, control strategy S2: continuous per-wheel differential-drive arcs — used for the reported lap-time results |

## Key design points

- **Video-level, not frame-level, splitting.** Adjacent frames are near-
  identical, so a random frame split would leak information between train
  and test. Splitting whole laps (4 train / 1 val / 1 test per direction)
  forces the model to generalise to unseen recording sessions.
- **Class imbalance handling.** A 24:1 ratio between the most and least
  common classes is addressed with class-weighted cross-entropy plus a
  `WeightedRandomSampler`; v3 softens the sampler weights to `w^0.3` to avoid
  overfitting the 168-frame `left` class.
- **Confidence gate + majority vote.** Non-`straight` predictions below 0.55
  confidence are relabelled `straight`, and the last 3 predictions are
  majority-voted before being sent to the motors, trading ~150 ms of latency
  for resistance to single-frame misclassifications.
- **Hardware-aware control.** The Yukon per-wheel API caps out at 19 Hz
  (52.6 ms/command, measured over 50 calls); the temporal filter's update
  rate was chosen to stay within this bound.

## Known limitation in this repo

`05_deploy_spin_burst.ipynb` and `06_deploy_differential_drive.ipynb` both
load `best_model_v3.pth`, but the report states v2 was used for all reported
deployment results (Sections V-B–V-E) because of its better `right`-class
recall. This is preserved as-is from the original lab notebooks rather than
silently edited — worth checking against your own run logs before treating
the S1/S2 notebooks as an exact reproduction of the reported numbers.

## Requirements

See [`requirements.txt`](requirements.txt). Frame extraction, labelling, and
training run with standard PyTorch/OpenCV. The two deployment notebooks
additionally need:
- ZED SDK + `pyzed` (proprietary, not on PyPI — install from
  [stereolabs.com](https://www.stereolabs.com/developers/))
- a `motors.py` module exposing the Yukon serial API used on this specific
  robot (hardware-specific, not included)

Trained weights (`best_model_v2.pth`, `best_model_v3.pth`, ~43 MB each) are
not committed to this repo — see [Releases] or track them with Git LFS if
you add them back in.

## Results in more depth

Per-corner deployment testing, the S1/S2 control comparison, the direction-
asymmetry analysis, and full methodology are in the [report](docs/report.pdf).
