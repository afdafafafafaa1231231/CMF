

## What is implemented
- Pose encoder accepting joint-wise **HRNet-W48-derived 256-dimensional** features for **17** COCO joints. Nine-frame joint-wise depthwise temporal convolution.
- Four **six-axis** IMU streams after external timestamp interpolation/median filtering; four residual TCN blocks of 128 channels, with dilations **1, 2, 4, 8**.
- Symmetric **4-head joint↔sensor attention** using common dimension **128**, yielding joint-sensor maps.
- Per-frame two-way adaptive modality gate, probability **0.2** modality dropout.
- **2-layer bidirectional LSTM**, per-direction hidden **256**, and dropout **0.3**.
- Four-way phase classification, phase-conditioned multi-class error classification, normalized severity head.
- Objective: error CE + **0.3 × phase CE + 0.1 × probability temporal smoothness**. Severity head is optional; manuscript does not specify supervised severity loss or weight.
- Per-subject splits, training, evaluation and 4 architecture ablations.

## Install & quick verification

```bash
pip install torch numpy
python smoke_test.py
python train.py --make-synthetic --data synthetic.npz --epochs 1 --batch-size 6 --out runs/smoke
```

The generated data are **synthetic smoke-test inputs**, not FitForm/GymAction/SportsPose and **must not** be used to claim the paper's published numbers.

## Training on real synchronized data

Prepare `.npz` with **fixed-length 64-frame clips** at 30 Hz:

| key | shape/dtype | meaning |
| --- | --- | --- |
| `pose` | `[N,64,17,256]` float32 | per-joint 256-d backbone features in fixed COCO order |
| `imu` | `[N,64,4,6]` float32 | four calibrated IMUs, ax/ay/az/gx/gy/gz, interpolated to video timestamps |
| `errors` | `[N,64]` int64 | error class index (0 normal; other IDs defined consistently by dataset) |
| `phases` | `[N,64]` int64 | 0 preparation, 1 eccentric, 2 concentric, 3 recovery |
| `subjects` | `[N]` str/int | participant identity, for leakage-free split |
| `severity` | `[N,64]` float32 (optional) | normalized error severity, only if reliable ground truth available |

To align an original 100-Hz IMU stream, use `data.resample_imu(imu, imu_times, video_times)` **after calibration**. Use `median_filter_imu(...)` to mitigate spikes. All sensors must follow a consistent body-placement ordering. Missing/noisy data should have explicitly recorded masks and preprocessing decisions.

```bash
python train.py --data /path/to/your_multimodal_64frame_clips.npz \
  --epochs 80 --batch-size 32 --lr 0.0001 --num-errors 5 --out runs/full
python train.py --data /path/to/your_multimodal_64frame_clips.npz \
  --ablation no_attention --out runs/no_attention
python train.py --data /path/to/your_multimodal_64frame_clips.npz \
  --ablation no_gate --out runs/no_gate
python train.py --data /path/to/your_multimodal_64frame_clips.npz \
  --ablation no_phase --out runs/no_phase
python train.py --data /path/to/your_multimodal_64frame_clips.npz \
  --ablation pose_only --out runs/pose_only
```

## Files

- `model.py` model, fusion, loss
- `data.py` NPZ interface, timestamp resampling, median filter, leakage-free partition, mock data
- `train.py` training, validation, checkpoint, test metrics
- `smoke_test.py` deterministic shape, gradient, ablation and data checks
