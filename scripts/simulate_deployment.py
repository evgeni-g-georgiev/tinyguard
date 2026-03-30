#!/usr/bin/env python3
"""
simulate_deployment.py — Deployment lifecycle simulation for AcousticEncoder.

Simulates the full on-device lifecycle for each of the 16 MIMII machines:

  Phase 0 — Training  : 10 min of normal audio → fit f_s SVDD, set threshold
  Rounds 1–3          : 5 min normal → 5 min abnormal  (repeated 3×)

No audio is reused across phases. Threshold is set from training scores
(deployment-realistic: the chip has no held-out normal data).

Each machine produces one timeline plot showing:
  - Anomaly scores over time
  - False positives (▼ orange): normal clips scored above threshold
  - First detection (★ green) + delay annotation per anomaly round
  - "Not detected" label where the model misses the anomaly window entirely

Usage
-----
    python scripts/simulate_deployment.py
    python scripts/simulate_deployment.py --checkpoint outputs/student/acoustic_encoder.pt
"""

import argparse
import glob
import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.cnn import AcousticEncoder
from src.models.separator import score_clips, train_fs

# ── config ────────────────────────────────────────────────────────────────────

DEFAULT_CKPT  = "outputs/student/acoustic_encoder.pt"
DATA_ROOT     = "data"
OUTPUT_DIR    = "outputs/deployment_sim"

SAMPLE_RATE   = 16_000
FRAME_LEN     = 15_600
N_FFT         = 1024
HOP_LENGTH    = 256
N_MELS        = 64
LOG_OFFSET    = 1e-6
CLIP_SECS     = 10.0          # MIMII nominal clip duration

TRAIN_MINS    = 10
MONITOR_MINS  = 5
N_ROUNDS      = 3
THRESHOLD_PCT = 95
SEED          = 42

TRAIN_CLIPS   = int(TRAIN_MINS   * 60 / CLIP_SECS)   # 60
MONITOR_CLIPS = int(MONITOR_MINS * 60 / CLIP_SECS)   # 30

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]


# ── audio helpers ─────────────────────────────────────────────────────────────

def log_mel(frame: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=frame, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0,
    )
    return np.log(mel + LOG_OFFSET)[np.newaxis, :, :]


def embed_clip(wav_path: str, model: AcousticEncoder, device) -> np.ndarray | None:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    n_frames  = len(audio) // FRAME_LEN
    if n_frames == 0:
        return None
    mels = np.stack([
        log_mel(audio[i * FRAME_LEN:(i + 1) * FRAME_LEN].astype(np.float32))
        for i in range(n_frames)
    ])
    with torch.no_grad():
        embs = model(torch.tensor(mels, dtype=torch.float32).to(device)).cpu().numpy()
    return embs


# ── simulation ────────────────────────────────────────────────────────────────

def simulate_machine(
    machine_type: str, machine_id: str, model: AcousticEncoder, device
) -> dict | None:

    base       = os.path.join(DATA_ROOT, machine_type, machine_id)
    norm_paths = sorted(glob.glob(os.path.join(base, "normal",   "*.wav")))
    anom_paths = sorted(glob.glob(os.path.join(base, "abnormal", "*.wav")))
    if not norm_paths or not anom_paths:
        return None

    rng        = np.random.default_rng(SEED)
    norm_paths = list(rng.permutation(norm_paths))
    anom_paths = list(rng.permutation(anom_paths))

    # Reduce rounds if the machine doesn't have enough clips
    n_rounds = min(
        N_ROUNDS,
        (len(norm_paths) - TRAIN_CLIPS) // MONITOR_CLIPS,
        len(anom_paths) // MONITOR_CLIPS,
    )
    if n_rounds <= 0 or len(norm_paths) < TRAIN_CLIPS:
        print("  Insufficient clips — skipping")
        return None

    # Allocate clips with no reuse
    train_paths = norm_paths[:TRAIN_CLIPS]
    norm_eval   = norm_paths[TRAIN_CLIPS : TRAIN_CLIPS + n_rounds * MONITOR_CLIPS]
    anom_eval   = anom_paths[:n_rounds * MONITOR_CLIPS]

    # ── Training phase ────────────────────────────────────────────────────────
    train_embs = []
    for path in tqdm(train_paths, desc="  training", leave=False, unit="clip"):
        embs = embed_clip(path, model, device)
        if embs is not None:
            train_embs.append(embs)
    if not train_embs:
        return None

    model_fs, centroid = train_fs(np.vstack(train_embs))
    train_scores       = score_clips(train_embs, model_fs, centroid)
    threshold          = float(np.percentile(train_scores, THRESHOLD_PCT))

    # ── Timeline construction ─────────────────────────────────────────────────
    # events:   list of {'t': minutes, 'score': float, 'phase': str, 'round': int}
    # segments: list of {'phase': str, 'round': int, 't_start': float, 't_end': float}
    events   = []
    segments = []
    t        = 0.0   # running time in seconds

    def add_segment(phase, round_idx, scores_list):
        nonlocal t
        seg_start = t
        for score in scores_list:
            events.append({"t": t / 60, "score": score, "phase": phase, "round": round_idx})
            t += CLIP_SECS
        segments.append({"phase": phase, "round": round_idx,
                          "t_start": seg_start / 60, "t_end": t / 60})

    add_segment("train", 0, train_scores)

    # ── Monitoring rounds ─────────────────────────────────────────────────────
    round_results = []
    for r in range(n_rounds):
        # Normal window
        norm_r_paths = norm_eval[r * MONITOR_CLIPS:(r + 1) * MONITOR_CLIPS]
        norm_embs_r  = []
        for path in tqdm(norm_r_paths, desc=f"  R{r+1} norm", leave=False, unit="clip"):
            embs = embed_clip(path, model, device)
            if embs is not None:
                norm_embs_r.append(embs)
        norm_scores_r = score_clips(norm_embs_r, model_fs, centroid) if norm_embs_r else []
        add_segment("normal", r + 1, norm_scores_r)

        # Anomaly window
        anom_r_paths = anom_eval[r * MONITOR_CLIPS:(r + 1) * MONITOR_CLIPS]
        anom_embs_r  = []
        for path in tqdm(anom_r_paths, desc=f"  R{r+1} anom", leave=False, unit="clip"):
            embs = embed_clip(path, model, device)
            if embs is not None:
                anom_embs_r.append(embs)
        anom_scores_r = score_clips(anom_embs_r, model_fs, centroid) if anom_embs_r else []
        add_segment("anomaly", r + 1, anom_scores_r)

        # Detection: first anomaly clip to exceed threshold
        detection_idx = None
        for i, score in enumerate(anom_scores_r):
            if score >= threshold:
                detection_idx = i
                break

        round_results.append({
            "round":                r + 1,
            "n_false_pos":          sum(1 for s in norm_scores_r if s >= threshold),
            "n_normal_clips":       len(norm_scores_r),
            "n_anom_clips":         len(anom_scores_r),
            "detected":             detection_idx is not None,
            "detection_delay_secs": int(detection_idx * CLIP_SECS) if detection_idx is not None else None,
            "detection_idx":        detection_idx,
        })

    return {
        "events":        events,
        "segments":      segments,
        "threshold":     threshold,
        "round_results": round_results,
        "n_rounds":      n_rounds,
        "fs_params":     sum(p.numel() for p in model_fs.parameters()),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_machine(result: dict, machine_type: str, machine_id: str) -> str:
    events    = result["events"]
    segments  = result["segments"]
    threshold = result["threshold"]
    fs_params = result["fs_params"]

    # Separate events by phase
    train_evts  = [e for e in events if e["phase"] == "train"]
    normal_evts = [e for e in events if e["phase"] == "normal"]
    anomaly_evts = [e for e in events if e["phase"] == "anomaly"]

    # Convert times to seconds
    def ts(e): return e["t"] * 60

    # ── Summary metrics for subtitle ─────────────────────────────────────────
    n_detected   = sum(1 for r in result["round_results"] if r["detected"])
    detection_pct = 100 * n_detected / result["n_rounds"]
    n_fps        = sum(r["n_false_pos"] for r in result["round_results"])
    n_norm_mon   = sum(r["n_normal_clips"] for r in result["round_results"])
    false_alarm_pct = 100 * n_fps / n_norm_mon if n_norm_mon > 0 else 0

    # AUC over monitoring normal + all anomaly clips
    try:
        auc_labels = [0] * len(normal_evts) + [1] * len(anomaly_evts)
        auc_scores = [e["score"] for e in normal_evts + anomaly_evts]
        auc = roc_auc_score(auc_labels, auc_scores)
    except Exception:
        auc = float("nan")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    # Background shading: light blue for training, light salmon for anomaly,
    # no shading for normal monitoring windows
    train_patched = False
    anom_patched  = False
    for seg in segments:
        if seg["phase"] == "train":
            ax.axvspan(seg["t_start"] * 60, seg["t_end"] * 60,
                       color="#dce9f5", alpha=0.8, zorder=0,
                       label="Training phase" if not train_patched else None)
            train_patched = True
        elif seg["phase"] == "anomaly":
            ax.axvspan(seg["t_start"] * 60, seg["t_end"] * 60,
                       color="#fde8e8", alpha=0.8, zorder=0,
                       label="Anomaly injection" if not anom_patched else None)
            anom_patched = True

    # Scatter: all normal clips (train + monitoring) in blue, anomaly in orange
    all_normal = train_evts + normal_evts
    if all_normal:
        ax.scatter([ts(e) for e in all_normal], [e["score"] for e in all_normal],
                   color="#5b9bd5", s=20, alpha=0.7, zorder=2, label="Normal clips")
    if anomaly_evts:
        ax.scatter([ts(e) for e in anomaly_evts], [e["score"] for e in anomaly_evts],
                   color="#f4a261", s=20, alpha=0.8, zorder=2, label="Anomaly clips")

    # Threshold line
    ax.axhline(threshold, color="#c0392b", linestyle="--", linewidth=1.3,
               label=f"Threshold ({threshold:.3f})", zorder=3)

    # ── Detection annotations ─────────────────────────────────────────────────
    for rr in result["round_results"]:
        anom_evts_r = [e for e in events
                       if e["phase"] == "anomaly" and e["round"] == rr["round"]]
        anom_seg    = next(s for s in segments
                          if s["phase"] == "anomaly" and s["round"] == rr["round"])

        if rr["detected"] and rr["detection_idx"] < len(anom_evts_r):
            idx          = rr["detection_idx"]
            detect_t_s   = ts(anom_evts_r[idx])
            detect_score = anom_evts_r[idx]["score"]
            delay        = rr["detection_delay_secs"]
            ax.annotate(
                f"Detected!\n({delay}s delay)",
                xy=(detect_t_s, detect_score),
                xytext=(30, 25), textcoords="offset points",
                color="#c0392b", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
                zorder=5,
            )
        else:
            mid_t_s = (anom_seg["t_start"] + anom_seg["t_end"]) / 2 * 60
            ax.text(mid_t_s, threshold * 0.6, "Not detected",
                    ha="center", va="center", fontsize=8,
                    color="#c0392b", style="italic", alpha=0.85)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Anomaly score", fontsize=10)
    ax.set_xlim(left=0)
    ax.margins(y=0.2)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    node_label = f"{machine_type.capitalize()} {machine_id.replace('_', ' ')}"
    ax.set_title(
        f"Node: {node_label}\n"
        f"Detection={detection_pct:.0f}%  |  "
        f"False alarm={false_alarm_pct:.1f}%  |  "
        f"AUC={auc:.3f}",
        fontsize=9, color="#333333",
    )
    fig.suptitle(
        "TinyML Deployment Simulation — Anomalous Sound Detection\n"
        f"Arduino Nano 33 BLE  |  SVDD ({fs_params} params)  |  AcousticEncoder embeddings",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{machine_type}_{machine_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Run:  python scripts/train_student.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = AcousticEncoder()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded: epoch {ckpt['epoch']}, val MSE {ckpt['val_loss']:.5f}")
    print(f"\nSimulation: {TRAIN_MINS} min training → "
          f"{N_ROUNDS}× [{MONITOR_MINS} min normal + {MONITOR_MINS} min anomaly]")
    print(f"Clips:      {TRAIN_CLIPS} train  |  {MONITOR_CLIPS}/window  |  seed={SEED}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            key = f"{mtype}/{mid}"
            print(f"[{key}]")
            result = simulate_machine(mtype, mid, model, device)
            if result is None:
                print("  skipped\n")
                continue

            plot_path = plot_machine(result, mtype, mid)
            all_results[key] = {
                "threshold": round(result["threshold"], 5),
                "n_rounds":  result["n_rounds"],
                "rounds":    result["round_results"],
            }
            print(f"  Plot → {plot_path}")
            for rr in result["round_results"]:
                delay = (f"{rr['detection_delay_secs']}s"
                         if rr["detected"] else "not detected")
                print(f"  Round {rr['round']}: "
                      f"FP={rr['n_false_pos']}/{rr['n_normal_clips']}  "
                      f"Delay={delay}")
            print()

    if not all_results:
        print("No results — check data paths.")
        return

    # ── Overall summary ───────────────────────────────────────────────────────
    all_rounds    = [r for res in all_results.values() for r in res["rounds"]]
    total_fp      = sum(r["n_false_pos"] for r in all_rounds)
    total_norm    = sum(r["n_normal_clips"] for r in all_rounds)
    n_detected    = sum(1 for r in all_rounds if r["detected"])
    total_rounds  = len(all_rounds)
    delays        = [r["detection_delay_secs"] for r in all_rounds if r["detected"]]
    mean_delay    = float(np.mean(delays)) if delays else float("nan")

    print(f"{'─' * 55}")
    print(f"  Machines evaluated  : {len(all_results)}")
    print(f"  Total normal clips  : {total_norm}  |  FP: {total_fp}  "
          f"({100*total_fp/total_norm:.1f}%)")
    print(f"  Detection rate      : {n_detected}/{total_rounds} rounds "
          f"({100*n_detected/total_rounds:.0f}%)")
    print(f"  Mean detection delay: {mean_delay:.0f}s")
    print(f"{'─' * 55}")
    print(f"\nPlots   → {OUTPUT_DIR}/*.png")
    print(f"Results → {OUTPUT_DIR}/results.yaml")

    with open(os.path.join(OUTPUT_DIR, "results.yaml"), "w") as f:
        yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
