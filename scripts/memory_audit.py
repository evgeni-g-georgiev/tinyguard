#!/usr/bin/env python3
"""
memory_audit.py — Flash and SRAM budget audit for the full on-device pipeline.

Run immediately after implementing AcousticEncoder (before training) to catch
architecture problems early. Re-run after TFLite conversion for real file size.

Covers every memory consumer in both on-device phases:
  Phase 1 — f_s training  (first 10 minutes: encoder inference + SVDD backprop)
  Phase 2 — inference     (continuous monitoring: encoder + SVDD forward only)

Usage
-----
    python scripts/memory_audit.py                        # static audit
    python scripts/memory_audit.py --tflite path/to.tflite  # + real file size
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.cnn import AcousticEncoder

# ── hardware limits ─────────────────────────────────────────────────────────

FLASH_BUDGET_KB = 2048
SRAM_BUDGET_KB  = 756

# ── activation trace (INT8 bytes, one sample) ───────────────────────────────
# Each row: (label, in_shape, out_shape)
# Peak concurrent = prod(in_shape) + prod(out_shape)  ← both tensors live at once

ACTIVATION_TRACE = [
    ("Stem",  (1, 1,   64, 61), (1, 32,  32, 31)),
    ("B1 DW", (1, 32,  32, 31), (1, 32,  16, 16)),   # ← SRAM bottleneck
    ("B1 PW", (1, 32,  16, 16), (1, 64,  16, 16)),
    ("B2 DW", (1, 64,  16, 16), (1, 64,   8,  8)),
    ("B2 PW", (1, 64,   8,  8), (1, 128,  8,  8)),
    ("B3 DW", (1, 128,  8,  8), (1, 128,  8,  8)),
    ("B3 PW", (1, 128,  8,  8), (1, 128,  8,  8)),
    ("B4 DW", (1, 128,  8,  8), (1, 128,  4,  4)),
    ("B4 PW", (1, 128,  4,  4), (1, 256,  4,  4)),
    ("B5 DW", (1, 256,  4,  4), (1, 256,  4,  4)),
    ("B5 PW", (1, 256,  4,  4), (1, 256,  4,  4)),
    ("B6 DW", (1, 256,  4,  4), (1, 256,  2,  2)),
    ("B6 PW", (1, 256,  2,  2), (1, 512,  2,  2)),
    ("B7 DW", (1, 512,  2,  2), (1, 512,  2,  2)),
    ("B7 PW", (1, 512,  2,  2), (1, 512,  2,  2)),
    ("Pool",  (1, 512,  2,  2), (1, 512,  1,  1)),
    ("Head",  (1, 512,       ), (1,  32,       )),
]


# ── helpers ─────────────────────────────────────────────────────────────────

def prod(shape):
    r = 1
    for s in shape:
        r *= s
    return r


def kb(n_bytes):
    return n_bytes / 1024


def bar(used_kb, budget_kb, width=30):
    frac = min(used_kb / budget_kb, 1.0)
    filled = int(frac * width)
    colour = "\033[91m" if frac > 0.85 else "\033[93m" if frac > 0.6 else "\033[92m"
    reset  = "\033[0m"
    return f"{colour}{'█' * filled}{'░' * (width - filled)}{reset} {frac:.0%}"


# ── flash audit ─────────────────────────────────────────────────────────────

def audit_flash(model: nn.Module):
    """
    Compute TFLite INT8 model flash footprint layer by layer.

    After BN folding (which TFLite does on export):
      - Conv/Linear weights → INT8 (1 byte each)
      - One INT32 bias per output channel (mandatory for INT8 TFLite conv)
      - One float32 per-channel quant scale per output channel
    BN layers disappear entirely (absorbed into conv weights + biases).
    """
    int8_total   = 0
    bias_total   = 0
    quant_total  = 0
    rows         = []

    for name, m in model.named_modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue
        out_ch      = m.weight.shape[0]
        int8_bytes  = m.weight.numel()          # 1 byte each after quantisation
        bias_bytes  = out_ch * 4                # one INT32 per output channel
        quant_bytes = out_ch * 4                # one float32 scale per output channel
        rows.append((name, int8_bytes, bias_bytes, quant_bytes))
        int8_total  += int8_bytes
        bias_total  += bias_bytes
        quant_total += quant_bytes

    return rows, int8_total, bias_total, quant_total


# ── SRAM activation trace ────────────────────────────────────────────────────

def audit_activations():
    """Trace peak concurrent activation bytes (INT8) through the network."""
    peak      = 0
    peak_op   = ""
    act_rows  = []

    for label, in_shape, out_shape in ACTIVATION_TRACE:
        in_bytes  = prod(in_shape)
        out_bytes = prod(out_shape)
        concurrent = in_bytes + out_bytes
        act_rows.append((label, in_shape, out_shape, in_bytes, out_bytes, concurrent))
        if concurrent > peak:
            peak    = concurrent
            peak_op = label

    return act_rows, peak, peak_op


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", metavar="PATH",
                        help="Path to exported .tflite file for real size check")
    args = parser.parse_args()

    W = 65
    print("═" * W)
    print("  AcousticEncoder — Memory Audit")
    print("═" * W)

    model      = AcousticEncoder()
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Parameters : {total_params:,}  ({trainable:,} trainable)")
    print(f"  float32    : {kb(total_params * 4):.0f} KB  (GPU training footprint — not on device)")

    # ── Flash ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  FLASH BUDGET  ({FLASH_BUDGET_KB:,} KB available)")
    print(f"{'─' * W}")
    print(f"  {'Layer':<35} {'INT8 wts':>9}  {'INT32 bias':>10}  {'Quant':>8}")
    print(f"  {'─'*35}  {'─'*9}  {'─'*10}  {'─'*8}")

    rows, int8_total, bias_total, quant_total = audit_flash(model)
    for name, i8, b, q in rows:
        short = name[-34:] if len(name) > 34 else name
        print(f"  {short:<35} {i8:>8}B  {b:>9}B  {q:>7}B")

    TFLITE_META_KB  = 10     # flatbuffer graph/tensor descriptors
    APP_CODE_KB     = 50     # Arduino sketch estimate
    model_kb        = kb(int8_total + bias_total + quant_total)
    total_flash_kb  = model_kb + TFLITE_META_KB + APP_CODE_KB

    print(f"  {'─'*35}  {'─'*9}  {'─'*10}  {'─'*8}")
    print(f"  {'INT8 weights':<35} {kb(int8_total):>8.1f}KB")
    print(f"  {'INT32 biases':<35} {kb(bias_total):>8.1f}KB")
    print(f"  {'Per-channel quant scales':<35} {kb(quant_total):>8.1f}KB")
    print(f"  {'TFLite flatbuffer metadata':<35} {'~'+str(TFLITE_META_KB):>8}KB")
    print(f"  {'Application code (estimate)':<35} {'~'+str(APP_CODE_KB):>8}KB")
    print(f"  {'─'*35}")
    print(f"  {'TOTAL':<35} {total_flash_kb:>8.1f}KB  /  {FLASH_BUDGET_KB} KB")
    print(f"  {bar(total_flash_kb, FLASH_BUDGET_KB)}")
    status = "✓ PASS" if total_flash_kb < FLASH_BUDGET_KB else "✗ FAIL"
    print(f"  {status}  —  {FLASH_BUDGET_KB - total_flash_kb:.0f} KB headroom")

    # Real TFLite file size if provided
    if args.tflite:
        if os.path.exists(args.tflite):
            real_kb = os.path.getsize(args.tflite) / 1024
            print(f"\n  Actual .tflite file : {real_kb:.1f} KB")
            status = "✓ PASS" if real_kb + APP_CODE_KB < FLASH_BUDGET_KB else "✗ FAIL"
            print(f"  {status} (model + app code = {real_kb + APP_CODE_KB:.1f} KB)")
        else:
            print(f"\n  WARNING: {args.tflite} not found")

    # ── SRAM — activation peak ──────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print(f"  SRAM — ACTIVATION PEAK TRACE (INT8, one inference)")
    print(f"{'─' * W}")
    print(f"  {'Op':<8}  {'Input shape':<20}  {'Output shape':<20}  {'Peak bytes':>10}")
    print(f"  {'─'*8}  {'─'*20}  {'─'*20}  {'─'*10}")

    act_rows, act_peak_bytes, peak_op = audit_activations()
    for label, in_s, out_s, in_b, out_b, conc in act_rows:
        marker = " ◄" if label == peak_op else ""
        print(f"  {label:<8}  {str(in_s):<20}  {str(out_s):<20}  {conc:>9}B{marker}")

    # Arena estimate: peak activations + scratch buffers + tail metadata
    SCRATCH_KB   = 20     # op scratch buffers (CMSIS-NN depthwise conv)
    METADATA_KB  = 5      # TfLiteEvalTensor structs, NodeAndRegistration, allocator
    arena_min_kb = kb(act_peak_bytes) + SCRATCH_KB + METADATA_KB
    arena_max_kb = arena_min_kb + 10  # alignment overhead

    print(f"\n  Peak concurrent activations : {act_peak_bytes:,} B = {kb(act_peak_bytes):.1f} KB  [{peak_op}]")
    print(f"  + scratch buffers (est.)    : ~{SCRATCH_KB} KB")
    print(f"  + tail metadata (est.)      : ~{METADATA_KB} KB")
    print(f"  Arena estimate              : ~{arena_min_kb:.0f}–{arena_max_kb:.0f} KB  (INT8)")
    print(f"  Arena at float32            : ~{arena_min_kb*4:.0f}–{arena_max_kb*4:.0f} KB  (if not quantised)")
    print(f"\n  ⚠  Arena must be verified empirically with RecordingMicroInterpreter")
    print(f"     after TFLite export — arena_used_bytes() excludes scratch buffers.")

    # ── SRAM — full budget ──────────────────────────────────────────────────
    AUDIO_BUF_KB    = 15_600 * 4 / 1024        # 61.0 KB
    MEL_BUF_KB      = 1 * 64 * 61 * 4 / 1024  # 15.3 KB
    # f_s SVDD: Linear(32→32, bias=True) + Linear(32→8, bias=False) = 1,312 params
    FS_WEIGHTS_KB   = 1312 * 4 / 1024          #  5.1 KB
    FS_GRADS_KB     = 1312 * 4 / 1024          #  5.1 KB  (training only)
    FS_ACTIVATIONS  = (32+32+32+8+8) * 4 / 1024 #  0.4 KB  (input,z1,a1,z2,a2 for backprop)
    CENTROID_KB     = 8 * 4 / 1024             #  0.03 KB (SVDD output dim still 8)
    BATCH_BUF_KB    = 12 * 32 * 4 / 1024       #  1.5 KB  (mini-batch 32D embeddings)
    STACK_KB        = 16

    inference_kb = arena_max_kb + AUDIO_BUF_KB + MEL_BUF_KB + FS_WEIGHTS_KB + CENTROID_KB + STACK_KB
    training_kb  = inference_kb + FS_GRADS_KB + FS_ACTIVATIONS + BATCH_BUF_KB

    print(f"\n{'─' * W}")
    print(f"  SRAM BUDGET  ({SRAM_BUDGET_KB} KB available)")
    print(f"{'─' * W}")
    print(f"  {'Component':<35} {'Inference':>10}  {'Training':>10}")
    print(f"  {'─'*35}  {'─'*10}  {'─'*10}")
    print(f"  {'TFLite Micro arena (INT8)':<35} {arena_max_kb:>9.1f}K  {arena_max_kb:>9.1f}K")
    print(f"  {'Audio capture buffer':<35} {AUDIO_BUF_KB:>9.1f}K  {AUDIO_BUF_KB:>9.1f}K")
    print(f"  {'Mel spectrogram buffer':<35} {MEL_BUF_KB:>9.1f}K  {MEL_BUF_KB:>9.1f}K")
    print(f"  {'f_s weights + centroid':<35} {FS_WEIGHTS_KB+CENTROID_KB:>9.1f}K  {FS_WEIGHTS_KB+CENTROID_KB:>9.1f}K")
    print(f"  {'f_s gradients (SGD, no mom.)':<35} {'—':>10}  {FS_GRADS_KB:>9.1f}K")
    print(f"  {'f_s forward activations':<35} {'—':>10}  {FS_ACTIVATIONS:>9.2f}K")
    print(f"  {'Mini-batch embedding buffer':<35} {'—':>10}  {BATCH_BUF_KB:>9.2f}K")
    print(f"  {'Stack + misc':<35} {STACK_KB:>9}K  {STACK_KB:>9}K")
    print(f"  {'─'*35}  {'─'*10}  {'─'*10}")
    print(f"  {'TOTAL':<35} {inference_kb:>9.1f}K  {training_kb:>9.1f}K")
    print(f"  {'Budget':<35} {SRAM_BUDGET_KB:>9}K  {SRAM_BUDGET_KB:>9}K")

    inf_status  = "✓ PASS" if inference_kb < SRAM_BUDGET_KB else "✗ FAIL"
    train_status = "✓ PASS" if training_kb  < SRAM_BUDGET_KB else "✗ FAIL"
    print(f"\n  Inference  {bar(inference_kb,  SRAM_BUDGET_KB)}  {inf_status}")
    print(f"  Training   {bar(training_kb,   SRAM_BUDGET_KB)}  {train_status}")

    print(f"\n  Note: SGD has no optimizer state (no momentum/Adam m,v vectors).")
    print(f"        If you switch to Adam for f_s, add {FS_GRADS_KB*2:.1f} KB for m+v buffers.")
    print("═" * W)


if __name__ == "__main__":
    main()
