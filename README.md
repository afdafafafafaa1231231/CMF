# CMF-ErrNet: executable method-level reproduction

Based on the user-provided manuscript **“Spatio-Temporal Multimodal Fusion for Exercise Error Recognition: A Cross-Modal Attention Framework Integrating Video Pose Estimation and Inertial Measurement.”**

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

## Key limitations vs. an exact paper replication

1. The manuscript specifies HRNet-W48 COCO-pretrained fine-tuning on RGB frames but does not supply model weights, dataset-specific fine-tuning code, or a precise definition of **per-joint 256-d heatmap-derived embeddings**. This repo starts at **extracted joint features**, not raw RGB pixels. Thus no claim of end-to-end HRNet replication is made.
2. Paper's `DepthwiseConv3D` is underspecified on its given `[T,J,256]` representation. We use **depthwise temporal Conv1d** on each joint feature channel with a 9-frame kernel.
3. The paper's residual cross attention formula adds a 128-d vector directly to a 256-d pose feature. We explicitly project pose to 128 dims **before residual addition** to resolve the mismatch, and implement the reverse sensor→joint attention described in prose.
4. The paper's gate equations use a 2-vector context and a second 2×2 projection. A compact MLP directly outputs the **two gate logits**; mathematically equivalent in expressiveness with a learned final linear layer.
5. Instead of non-differentiable `argmax` phase labels inside training, phase conditioning uses **soft phase probabilities**; inference can still use predicted phases. This allows the error loss to backpropagate to the phase head.
6. The original description names a severity regression head but **does not provide severity annotations or its training objective**. We output a [0,1] severity prediction but default to **zero severity-loss weight**; scores are uncalibrated without labels.
7. The benchmark sources named in the manuscript do **not automatically constitute matched RGB+4-IMU+per-frame error/phase records**. A paired, labeled dataset with the same exercise and participant on both streams is required. Dataset conversion and numbers in Tables 1–3 cannot be validated from the manuscript alone.
8. Exact error label taxonomy, sensor locations, clock calibration, zero-phase clipping policy, local calibration transformation, and train/test subject IDs are not specified. Choices here are transparent implementation assumptions.
9. Evaluation gives frame-wise macro metrics. Manuscript refers to clip accuracy and binary ROC AUC, but does not define a clip aggregation rule or map of error classes to positive labels. Define these per dataset before comparing paper tables.

## Files

- `model.py` model, fusion, loss
- `data.py` NPZ interface, timestamp resampling, median filter, leakage-free partition, mock data
- `train.py` training, validation, checkpoint, test metrics
- `smoke_test.py` deterministic shape, gradient, ablation and data checks
