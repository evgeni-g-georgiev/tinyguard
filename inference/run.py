#!/usr/bin/env python3
"""
run.py — Simulate the on-device monitoring lifecycle for all 16 MIMII machines.

Loads pre-trained separator artefacts (from separator/train.py) and runs the
deployment simulation: for each machine, 3 rounds of 5 min normal monitoring
followed by 5 min anomaly injection. The threshold is fixed from training —
no information from future clips is used.

For each machine, produces a timeline plot showing anomaly scores, false
positives, and detection events. Aggregates detection rate, false alarm rate,
and detection delay across all machines and rounds.

Inputs
------
  outputs/student/acoustic_encoder.pt        frozen AcousticEncoder
  outputs/separator/{mtype}_{mid}.pt         trained SVDD artefacts (×16)
  outputs/mimii_splits/splits.json           clip manifest
  data/mimii/                                MIMII WAV files

Outputs
-------
  outputs/inference/results.yaml             aggregate metrics
  outputs/inference/{mtype}_{mid}.png        per-machine timeline plots

Usage
-----
    python inference/run.py
    python inference/run.py --checkpoint outputs/student/acoustic_encoder.pt
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    STUDENT_DIR, SEPARATOR_DIR, MIMII_SPLITS, INFERENCE_DIR, MIMII_ROOT,
    MACHINE_TYPES, MACHINE_IDS,
    SAMPLE_RATE, FRAME_LEN, N_FFT, HOP_LENGTH, N_MELS, LOG_OFFSET,
    CLIP_SECS,
)
from distillation.cnn import AcousticEncoder
from separator.separator import FsSeparator, score_clips


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


# ── per-machine simulation ────────────────────────────────────────────────────

def run_machine(
    mtype: str, mid: str,
    enc: AcousticEncoder, device,
    splits: dict,
) -> dict | None:
    key = f"{mtype}/{mid}"

    artefact_path = SEPARATOR_DIR / f"{mtype}_{mid}.pt"
    if not artefact_path.exists():
        print(f"  [{key}] artefact not found at {artefact_path} — skipping.")
        return None

    if key not in splits:
        print(f"  [{key}] not in splits manifest — skipping.")
        return None

    # Load separator artefact
    art = torch.load(artefact_path, map_location="cpu", weights_only=False)
    model_fs = FsSeparator(
        input_dim=art.get("input_dim", 32),
        hidden_dim=art.get("hidden_dim", 32),
        output_dim=art.get("output_dim", 8),
    )
    model_fs.load_state_dict(art["state_dict"])
    model_fs.eval()
    centroid  = art["centroid"]
    threshold = art["threshold"]
    fs_params = art.get("n_params", sum(p.numel() for p in model_fs.parameters()))

    # Load clip paths from manifest
    test_normal_paths   = [str(MIMII_ROOT / p) for p in splits[key]["test_normal"]]
    test_abnormal_paths = [str(MIMII_ROOT / p) for p in splits[key]["test_abnormal"]]
    n_rounds = splits[key]["n_rounds"]

    # ── Timeline construction ──────────────────────────────────────────────
    events   = []   # {t, score, phase, round}
    segments = []   # {phase, round, t_start, t_end}
    t        = 0.0  # running time in seconds

    monitor_clips = len(test_normal_paths) // n_rounds

    def add_segment(phase, round_idx, wav_paths):
        nonlocal t
        seg_start = t
        emb_list  = []
        for path in tqdm(wav_paths, desc=f"    {phase[:4]} r{round_idx}", leave=False, unit="clip"):
            embs = embed_clip(path, enc, device)
            if embs is not None:
                emb_list.append(embs)
        scores = score_clips(emb_list, model_fs, centroid) if emb_list else []
        for score in scores:
            events.append({"t": t / 60, "score": score, "phase": phase, "round": round_idx})
            t += CLIP_SECS
        segments.append({"phase": phase, "round": round_idx,
                          "t_start": seg_start / 60, "t_end": t / 60})
        return scores

    # ── Monitoring rounds ──────────────────────────────────────────────────
    round_results = []
    for r in range(n_rounds):
        norm_paths = test_normal_paths[r * monitor_clips:(r + 1) * monitor_clips]
        anom_paths = test_abnormal_paths[r * monitor_clips:(r + 1) * monitor_clips]

        norm_scores = add_segment("normal",  r + 1, norm_paths)
        anom_scores = add_segment("anomaly", r + 1, anom_paths)

        detection_idx = next(
            (i for i, s in enumerate(anom_scores) if s >= threshold), None
        )
        round_results.append({
            "round":                r + 1,
            "n_false_pos":          sum(1 for s in norm_scores if s >= threshold),
            "n_normal_clips":       len(norm_scores),
            "n_anom_clips":         len(anom_scores),
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
        "fs_params":     fs_params,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_machine(result: dict, mtype: str, mid: str) -> str:
    events    = result["events"]
    segments  = result["segments"]
    threshold = result["threshold"]
    fs_params = result["fs_params"]

    train_evts   = [e for e in events if e["phase"] == "normal"]
    anomaly_evts = [e for e in events if e["phase"] == "anomaly"]

    def ts(e): return e["t"] * 60

    n_detected      = sum(1 for r in result["round_results"] if r["detected"])
    detection_pct   = 100 * n_detected / result["n_rounds"]
    n_fps           = sum(r["n_false_pos"] for r in result["round_results"])
    n_norm_mon      = sum(r["n_normal_clips"] for r in result["round_results"])
    false_alarm_pct = 100 * n_fps / n_norm_mon if n_norm_mon > 0 else 0

    try:
        auc_labels = [0] * len(train_evts) + [1] * len(anomaly_evts)
        auc_scores = [e["score"] for e in train_evts + anomaly_evts]
        auc = roc_auc_score(auc_labels, auc_scores)
    except Exception:
        auc = float("nan")

    fig, ax = plt.subplots(figsize=(14, 5))

    anom_patched = False
    for seg in segments:
        if seg["phase"] == "anomaly":
            ax.axvspan(seg["t_start"] * 60, seg["t_end"] * 60,
                       color="#fde8e8", alpha=0.8, zorder=0,
                       label="Anomaly injection" if not anom_patched else None)
            anom_patched = True

    if train_evts:
        ax.scatter([ts(e) for e in train_evts], [e["score"] for e in train_evts],
                   color="#5b9bd5", s=20, alpha=0.7, zorder=2, label="Normal clips")
    if anomaly_evts:
        ax.scatter([ts(e) for e in anomaly_evts], [e["score"] for e in anomaly_evts],
                   color="#f4a261", s=20, alpha=0.8, zorder=2, label="Anomaly clips")

    ax.axhline(threshold, color="#c0392b", linestyle="--", linewidth=1.3,
               label=f"Threshold ({threshold:.3f})", zorder=3)

    for rr in result["round_results"]:
        anom_evts_r = [e for e in events if e["phase"] == "anomaly" and e["round"] == rr["round"]]
        anom_seg    = next(s for s in segments if s["phase"] == "anomaly" and s["round"] == rr["round"])

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

    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Anomaly score", fontsize=10)
    ax.set_xlim(left=0)
    ax.margins(y=0.2)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    node_label = f"{mtype.capitalize()} {mid.replace('_', ' ')}"
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
    out_path = INFERENCE_DIR / f"{mtype}_{mid}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(STUDENT_DIR / "acoustic_encoder.pt"))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        print("Run:  python distillation/train.py")
        sys.exit(1)

    if not MIMII_SPLITS.exists():
        print(f"ERROR: Splits manifest not found at {MIMII_SPLITS}")
        print("Run:  python preprocessing/split_mimii.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc  = AcousticEncoder()
    enc.load_state_dict(ckpt["model_state_dict"])
    enc.to(device).eval()
    print(f"Loaded AcousticEncoder: epoch {ckpt['epoch']}, val MSE {ckpt['val_loss']:.5f}")

    with open(MIMII_SPLITS) as f:
        splits = json.load(f)

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for mtype in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            key = f"{mtype}/{mid}"
            print(f"[{key}]")
            result = run_machine(mtype, mid, enc, device, splits)
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
        print("No results — check that separator/train.py has been run and MIMII data is present.")
        return

    # ── Overall summary ───────────────────────────────────────────────────────
    all_rounds   = [r for res in all_results.values() for r in res["rounds"]]
    total_fp     = sum(r["n_false_pos"] for r in all_rounds)
    total_norm   = sum(r["n_normal_clips"] for r in all_rounds)
    n_detected   = sum(1 for r in all_rounds if r["detected"])
    total_rounds = len(all_rounds)
    delays       = [r["detection_delay_secs"] for r in all_rounds if r["detected"]]
    mean_delay   = float(np.mean(delays)) if delays else float("nan")
    median_delay = float(np.median(delays)) if delays else float("nan")
    max_delay    = int(max(delays)) if delays else None

    print(f"{'─' * 55}")
    print(f"  Machines evaluated  : {len(all_results)}")
    print(f"  Total normal clips  : {total_norm}  |  FP: {total_fp}  "
          f"({100*total_fp/total_norm:.1f}%)")
    print(f"  Detection rate      : {n_detected}/{total_rounds} rounds "
          f"({100*n_detected/total_rounds:.0f}%)")
    print(f"  Mean detection delay: {mean_delay:.0f}s  "
          f"(median {median_delay:.0f}s, max {max_delay}s)")
    print(f"{'─' * 55}")
    print(f"\nPlots   → {INFERENCE_DIR}/*.png")
    print(f"Results → {INFERENCE_DIR}/results.yaml")

    results_path = INFERENCE_DIR / "results.yaml"
    with open(results_path, "w") as f:
        yaml.dump(all_results, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
