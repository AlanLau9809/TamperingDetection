#!/usr/bin/env python3
"""
review_evaluate.py — Review All Evaluation Results
====================================================
Reads existing evaluation CSV files from:
    TamperingDetection/Evaluation Results/{EXP}_{MODEL_TAG}/

For every experiment (E1/E2/E3) and every model tag found, writes a review TXT
into:
    TamperingDetection/Evaluation Results/Review Evaluation/{MODEL_TAG}-Review/
        E1_{MODEL_TAG}-Review.txt  (4-class + 2-class binary analysis)
        E2_{MODEL_TAG}-Review.txt  (Covered+Defocused only metrics + binary)
        E3_{MODEL_TAG}-Review.txt  (4-class + 2-class binary analysis)

Also prints a ranked terminal summary of all models by smoothed 2-class performance.

2-class analysis follows the original UHCTD paper:
  • No retraining — group 4-class predictions: {Covered, Defocused, Moved} → Tampered
  • Metrics: TP, TN, FP, FN, TPR, FPR, Acc, hFAR (hourly false alarm rate)
  • hFAR = (FP / Normal_frames) * (FPS * 3600)
  • For E2: Moved frames excluded, but Normal IS included (to compute hFAR)
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support
)

# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Normal", 1: "Covered", 2: "Defocused", 3: "Moved"}
DAYS        = ["Day 3", "Day 4", "Day 5", "Day 6"]
FPS_CAM_A   = 3.0   # Camera A recording FPS

EXPERIMENTS = {
    "E1": {"name": "Cam A Train → Cam A Test",   "test_cam": "Camera A"},
    "E2": {"name": "Cam B Train → Cam A Test",   "test_cam": "Camera A"},
    "E3": {"name": "Cam A+B Train → Cam A Test", "test_cam": "Camera A"},
}

def _infer_config(tag: str) -> str:
    t = tag.upper()
    if "REFDIFF" in t: return "SlowFast-main/configs/UHCTD/SLOWFAST_UHCTD_REFDIFF.yaml"
    if "PLACES365" in t: return "SlowFast-main/configs/UHCTD/SLOWFAST_UHCTD_Places365.yaml"
    return "SlowFast-main/configs/UHCTD/SLOWFAST_UHCTD_RGB.yaml"

W = 60

def _sep(c="="): return c * W

# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_experiment_folders(eval_root):
    model_map = {}
    exclude = {"Analysis Plots", "Review Evaluation"}
    pat = re.compile(r'^(E[123])_(.+)$')
    for entry in sorted(os.listdir(eval_root)):
        if entry in exclude: continue
        folder = os.path.join(eval_root, entry)
        if not os.path.isdir(folder): continue
        m = pat.match(entry)
        if m:
            model_map.setdefault(m.group(2), []).append((m.group(1), folder))
    return model_map


def find_csv_for_day(folder, exp_key, model_tag, day):
    candidates = [f for f in os.listdir(folder)
                  if f.endswith(".csv") and day in f]
    if not candidates: return None
    if len(candidates) == 1: return os.path.join(folder, candidates[0])
    preferred = f"eval_{exp_key}_{model_tag}_cam_a_{day}.csv"
    return os.path.join(folder, preferred if preferred in candidates else sorted(candidates)[0])


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — 4-class
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, labels=None):
    if labels is None: labels = [0, 1, 2, 3]
    mask = np.isin(y_true, labels)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return _empty_metrics(labels)
    acc = accuracy_score(yt, yp)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        yt, yp, average='macro', zero_division=0, labels=labels)
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
        yt, yp, average=None, zero_division=0, labels=labels)
    cm = confusion_matrix(yt, yp, labels=labels)
    total = len(yt)
    tpr_c = np.zeros(4); fpr_c = np.zeros(4); acc_c = np.zeros(4)
    for idx, cls in enumerate(labels):
        tp = cm[idx, idx]; fn = cm[idx].sum() - tp
        fp = cm[:, idx].sum() - tp; tn = total - tp - fn - fp
        tpr_c[cls] = tp / (tp + fn) if (tp + fn) else 0.
        fpr_c[cls] = fp / (fp + tn) if (fp + tn) else 0.
        acc_c[cls] = (tp + tn) / total if total else 0.
    # Expand per-class arrays to size-4
    p4 = np.zeros(4); r4 = np.zeros(4); f4 = np.zeros(4)
    for i, cls in enumerate(labels):
        p4[cls] = prec_c[i]; r4[cls] = rec_c[i]; f4[cls] = f1_c[i]
    return dict(accuracy=acc, macro_precision=prec_m, macro_recall=rec_m,
                macro_f1=f1_m, precision_per=p4, recall_per=r4, f1_per=f4,
                tpr_per=tpr_c, fpr_per=fpr_c, acc_per=acc_c,
                confusion_matrix=cm, eval_labels=labels,
                n_filtered=int(total), n_total=int(len(y_true)))


def _empty_metrics(labels):
    return dict(accuracy=0., macro_precision=0., macro_recall=0., macro_f1=0.,
                precision_per=np.zeros(4), recall_per=np.zeros(4), f1_per=np.zeros(4),
                tpr_per=np.zeros(4), fpr_per=np.zeros(4), acc_per=np.zeros(4),
                confusion_matrix=np.zeros((len(labels), len(labels)), dtype=int),
                eval_labels=labels, n_filtered=0, n_total=0)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — 2-class binary (Normal vs Tampered)
# ─────────────────────────────────────────────────────────────────────────────

def compute_binary_metrics(y_true, y_pred, fps=FPS_CAM_A, exclude_moved=False):
    """
    Group 4-class predictions into Normal (0) vs Tampered (1).
    No retraining — groups prediction labels only.

    exclude_moved: for E2, Moved (class 3) frames are removed from evaluation
                  (but Normal frames ARE kept to compute hFAR).
    """
    if exclude_moved:
        mask = y_true != 3
        yt, yp = y_true[mask], y_pred[mask]
    else:
        yt, yp = y_true.copy(), y_pred.copy()

    # Binary mapping: 0=Normal, anything else=Tampered(1)
    yt_bin = (yt != 0).astype(int)
    yp_bin = (yp != 0).astype(int)

    n_total  = len(yt_bin)
    n_normal = int((yt_bin == 0).sum())
    n_tamp   = int((yt_bin == 1).sum())

    if n_total == 0:
        return dict(tp=0, tn=0, fp=0, fn=0, tpr=0., fpr=0.,
                    acc=0., hfar=0., n_normal=0, n_tampered=0, total=0)

    cm   = confusion_matrix(yt_bin, yp_bin, labels=[0, 1])
    tn   = int(cm[0, 0]); fp = int(cm[0, 1])
    fn   = int(cm[1, 0]); tp = int(cm[1, 1])

    tpr  = tp / (tp + fn) if (tp + fn) else 0.
    fpr  = fp / (fp + tn) if (fp + tn) else 0.
    acc  = (tp + tn) / n_total

    # hFAR: false alarms per hour of NORMAL footage
    # seconds of normal footage = n_normal / fps
    # hFAR = FP / (n_normal / fps / 3600)  = FP * fps * 3600 / n_normal
    hfar = (fp * fps * 3600. / n_normal) if n_normal > 0 else float('inf')

    return dict(tp=tp, tn=tn, fp=fp, fn=fn, tpr=tpr, fpr=fpr,
                acc=acc, hfar=hfar, n_normal=n_normal, n_tampered=n_tamp,
                total=n_total)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_metrics_block(m, tag):
    labels = m.get("eval_labels", [0, 1, 2, 3])
    hdr    = "2-Class Results [Covered+Defocused]" if len(labels)==2 else "4-Class Results"
    lines  = [f"\n{hdr} [{tag}]:",
              f"   Overall Accuracy : {m['accuracy']:.3f}",
              f"   Macro P={m['macro_precision']:.3f}  R={m['macro_recall']:.3f}  F1={m['macro_f1']:.3f}"]
    for cls in labels:
        lines.append(
            f"   {CLASS_NAMES[cls]:<12}: P={m['precision_per'][cls]:.3f}  "
            f"R={m['recall_per'][cls]:.3f}  F1={m['f1_per'][cls]:.3f}  "
            f"TPR={m['tpr_per'][cls]:.3f}  FPR={m['fpr_per'][cls]:.3f}  "
            f"Acc={m['acc_per'][cls]:.3f}")
    lines += [f"   Confusion Matrix (True↓ | Pred→):", str(m['confusion_matrix']),
              f"   Classes: {[f'{k}={CLASS_NAMES[k]}' for k in labels]}"]
    return lines


def _fmt_detail_table(m, tag):
    labels = m.get("eval_labels", [0, 1, 2, 3])
    cm = m['confusion_matrix']; total = cm.sum()
    hdr = (f"  {'Class':<12}  {'TP':>8}  {'FP':>8}  {'FN':>8}  {'TN':>8}  "
           f"{'TPR':>6}  {'FPR':>6}  {'Prec':>6}  {'F1':>6}  {'Acc':>6}")
    lines = [f"\n{'─'*W}", f"Detailed Class Statistics [{tag}]", f"{'─'*W}", hdr,
             "  " + "-" * (len(hdr) - 2)]
    for idx, cls in enumerate(labels):
        tp = int(cm[idx,idx]); fn = int(cm[idx].sum()-tp)
        fp = int(cm[:,idx].sum()-tp); tn = int(total-tp-fn-fp)
        lines.append(
            f"  {CLASS_NAMES[cls]:<12}  {tp:>8,}  {fp:>8,}  {fn:>8,}  {tn:>8,}  "
            f"{m['tpr_per'][cls]:>6.3f}  {m['fpr_per'][cls]:>6.3f}  "
            f"{m['precision_per'][cls]:>6.3f}  {m['f1_per'][cls]:>6.3f}  "
            f"{m['acc_per'][cls]:>6.3f}")
    lines += [f"  {'─'*56}",
              f"  {'Overall':<12}  {'':>8}  {'':>8}  {'':>8}  {'':>8}  "
              f"{'':>6}  {'':>6}  {m['macro_precision']:>6.3f}  "
              f"{m['macro_f1']:>6.3f}  {m['accuracy']:>6.3f}"]
    return lines


def _fmt_binary_block(b_raw, b_sm, day):
    """Format the 2-class (Normal vs Tampered) binary analysis for one day."""
    lines = [
        f"\n{'─'*W}",
        f"2-Class Analysis [Normal vs Tampered] — {day}",
        f"(4-class predictions grouped post-hoc)",
        f"{'─'*W}",
        f"  {'Metric':<10}  {'Raw':>10}  {'Smoothed':>10}",
        f"  {'─'*34}",
        f"  {'Total':<10}  {b_raw['total']:>10,}  {b_sm['total']:>10,}",
        f"  {'Normal':<10}  {b_raw['n_normal']:>10,}  {b_sm['n_normal']:>10,}",
        f"  {'Tampered':<10}  {b_raw['n_tampered']:>10,}  {b_sm['n_tampered']:>10,}",
        f"  {'─'*34}",
        f"  {'TP':<10}  {b_raw['tp']:>10,}  {b_sm['tp']:>10,}",
        f"  {'TN':<10}  {b_raw['tn']:>10,}  {b_sm['tn']:>10,}",
        f"  {'FP':<10}  {b_raw['fp']:>10,}  {b_sm['fp']:>10,}",
        f"  {'FN':<10}  {b_raw['fn']:>10,}  {b_sm['fn']:>10,}",
        f"  {'─'*34}",
        f"  {'TPR':<10}  {b_raw['tpr']:>10.4f}  {b_sm['tpr']:>10.4f}",
        f"  {'FPR':<10}  {b_raw['fpr']:>10.4f}  {b_sm['fpr']:>10.4f}",
        f"  {'Acc':<10}  {b_raw['acc']:>10.4f}  {b_sm['acc']:>10.4f}",
        f"  {'hFAR':<10}  {b_raw['hfar']:>10.2f}  {b_sm['hfar']:>10.2f}",
        f"  (hFAR = false alarms per hour of normal footage at {FPS_CAM_A} FPS)",
    ]
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

def build_report(exp_key, model_tag, day_results):
    exp_info   = EXPERIMENTS[exp_key]
    is_e2      = (exp_key == "E2")
    ckpt       = f"SlowFast-main/checkpoint/{exp_key}_{model_tag}_best.pth"
    config     = _infer_config(model_tag)
    out_dir    = f"Evaluation Results/{exp_key}_{model_tag}"
    sum_labels = [1, 2] if is_e2 else [0, 1, 2, 3]

    lines = [
        "UHCTD Model Evaluation Script — Full Sliding Window Sweep",
        _sep("="),
        f"Experiment  : {exp_key} — {exp_info['name']}",
        f"Model tag   : {model_tag}",
        f"Checkpoint  : {ckpt}",
        f"Config      : {config}",
        f"Window/stride: 32/32",
        f"Output dir  : {out_dir}",
        _sep("="),
    ]
    if is_e2:
        lines += [
            "NOTE (Experiment 2): Following original paper methodology,",
            "  only Covered and Defocused frames are used for 4-class metrics.",
            "  Normal and Moved frames EXCLUDED from 4-class evaluation.",
            "  (Normal frames ARE included in 2-class binary analysis for hFAR.)",
            _sep("="),
        ]
    lines += [f"Loading trained SlowFast model…",
              f"  Loaded checkpoint : {ckpt}", "",
              f"Starting evaluation on {exp_info['test_cam']} testing videos…"]

    for dr in day_results:
        day = dr["day"]; gt = dr["gt_stats"]; n = gt["total"]
        raw_m = dr["raw_m"]; sm_m = dr["smoothed_m"]
        b_raw = dr["binary_raw"]; b_sm = dr["binary_sm"]
        lines.append(f"\n{'='*55}")
        lines.append(f"Evaluating {day}")
        if is_e2:
            ef = gt.get(1,0)+gt.get(2,0)
            lines += [f"  Ground truth (full video): {n:,} frames total",
                      f"  Evaluated (Covered+Defocused): {ef:,}",
                      f"  [Moved frames excluded from 4-class; Normal kept for hFAR]"]
            for c in [1,2]:
                cnt=gt.get(c,0); pct=100.*cnt/ef if ef else 0.
                lines.append(f"    {CLASS_NAMES[c]:<12}: {cnt:6,} ({pct:.1f}%)")
        else:
            lines.append(f"  Ground truth: {n:,} frames total")
            for c, cn in CLASS_NAMES.items():
                cnt=gt.get(c,0); pct=100.*cnt/n if n else 0.
                lines.append(f"    {cn:<12}: {cnt:6,} ({pct:.1f}%)")
        fps = dr.get("fps", FPS_CAM_A); nw = dr.get("n_windows", n//32)
        lines += [f"  Video  : video.avi",
                  f"  Frames : {n:,}, FPS={fps}, window=32, stride=32",
                  f"  Windows: {nw} (no overlap)"]
        lines += _fmt_metrics_block(raw_m, "Raw")
        lines += _fmt_metrics_block(sm_m,  "Smoothed")
        lines += _fmt_detail_table(raw_m,  "Raw")
        lines += _fmt_detail_table(sm_m,   "Smoothed")
        lines += _fmt_binary_block(b_raw, b_sm, day)
        lines.append(f"  Saved: {out_dir}/eval_{exp_key}_{model_tag}_cam_a_{day}.csv")

    # Overall 4-class summary
    lines += [f"\n{_sep('=')}", "OVERALL EVALUATION SUMMARY (4-Class)"]
    if is_e2: lines.append("(Covered+Defocused only — Moved & Normal excluded)")
    lines.append(_sep("="))
    for dr in day_results:
        day=dr["day"]; r=dr["raw_m"]; sm=dr["smoothed_m"]
        lines += [f"{day}:",
                  f"  Raw      — Acc={r['accuracy']:.3f}  P={r['macro_precision']:.3f}  "
                  f"R={r['macro_recall']:.3f}  F1={r['macro_f1']:.3f}",
                  f"  Smoothed — Acc={sm['accuracy']:.3f}  P={sm['macro_precision']:.3f}  "
                  f"R={sm['macro_recall']:.3f}  F1={sm['macro_f1']:.3f}",
                  f"  Recall per class (smoothed): " +
                  "  ".join(f"{CLASS_NAMES[i]}={sm['recall_per'][i]:.3f}" for i in sum_labels),
                  f"  FPR    per class (smoothed): " +
                  "  ".join(f"{CLASS_NAMES[i]}={sm['fpr_per'][i]:.3f}" for i in sum_labels)]

    lines.append(f"\n{_sep('-')}")
    lines.append("Average across all test days (smoothed, 4-class):")
    avg_acc = np.mean([dr["smoothed_m"]["accuracy"]       for dr in day_results])
    avg_p   = np.mean([dr["smoothed_m"]["macro_precision"] for dr in day_results])
    avg_r   = np.mean([dr["smoothed_m"]["macro_recall"]   for dr in day_results])
    avg_f1  = np.mean([dr["smoothed_m"]["macro_f1"]       for dr in day_results])
    lines.append(f"  Acc={avg_acc:.4f}  P={avg_p:.4f}  R={avg_r:.4f}  F1={avg_f1:.4f}")
    tpr_avg = np.mean([dr["smoothed_m"]["tpr_per"] for dr in day_results], axis=0)
    fpr_avg = np.mean([dr["smoothed_m"]["fpr_per"] for dr in day_results], axis=0)
    acc_avg = np.mean([dr["smoothed_m"]["acc_per"] for dr in day_results], axis=0)
    lines.append("  Average TPR: " + "  ".join(f"{CLASS_NAMES[i]}={tpr_avg[i]:.4f}" for i in sum_labels))
    lines.append("  Average FPR: " + "  ".join(f"{CLASS_NAMES[i]}={fpr_avg[i]:.4f}" for i in sum_labels))
    lines.append("  Average Acc: " + "  ".join(f"{CLASS_NAMES[i]}={acc_avg[i]:.4f}" for i in sum_labels))

    # Overall 2-class summary
    lines += [f"\n{_sep('=')}", "OVERALL 2-CLASS ANALYSIS SUMMARY [Normal vs Tampered]",
              "(4-class predictions grouped post-hoc)",
              _sep("=")]
    hdr2 = (f"  {'Day':<8}  {'Mode':<10}  {'TP':>8}  {'TN':>8}  {'FP':>8}  "
            f"{'FN':>8}  {'TPR':>7}  {'FPR':>7}  {'Acc':>7}  {'hFAR':>8}")
    lines += [hdr2, "  " + "─"*(len(hdr2)-2)]
    for dr in day_results:
        day=dr["day"]
        for lbl, b in [("Raw", dr["binary_raw"]), ("Smoothed", dr["binary_sm"])]:
            lines.append(
                f"  {day:<8}  {lbl:<10}  {b['tp']:>8,}  {b['tn']:>8,}  "
                f"{b['fp']:>8,}  {b['fn']:>8,}  "
                f"{b['tpr']:>7.4f}  {b['fpr']:>7.4f}  {b['acc']:>7.4f}  {b['hfar']:>8.2f}")
    lines.append(f"  {'─'*64}")
    avg_tpr  = np.mean([dr["binary_sm"]["tpr"]  for dr in day_results])
    avg_fpr  = np.mean([dr["binary_sm"]["fpr"]  for dr in day_results])
    avg_bacc = np.mean([dr["binary_sm"]["acc"]  for dr in day_results])
    avg_hfar = np.mean([dr["binary_sm"]["hfar"] for dr in day_results
                        if dr["binary_sm"]["hfar"] != float('inf')])
    lines.append(f"  {'Average':<8}  {'Smoothed':<10}  {'':>8}  {'':>8}  "
                 f"{'':>8}  {'':>8}  "
                 f"{avg_tpr:>7.4f}  {avg_fpr:>7.4f}  {avg_bacc:>7.4f}  {avg_hfar:>8.2f}")

    lines += [f"\nEvaluation complete!", f"Results in: {out_dir}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal ranking
# ─────────────────────────────────────────────────────────────────────────────

EXP_NAMES = {
    "E1": "Cam A Train → Cam A Test",
    "E2": "Cam B Train → Cam A Test",
    "E3": "Cam A+B Train → Cam A Test",
}


def print_ranking(all_scores):
    if not all_scores:
        return
    cw = max(len(s["model_tag"]) for s in all_scores) + 2

    # ── Overall Macro-F1 quick summary ─────────────────────────────────────
    srt = sorted(all_scores, key=lambda x: x["avg_f1"], reverse=True)
    print()
    print("═"*(cw+66))
    print("  OVERALL RANKING — Mean Smoothed Macro-F1 (4-class, all days & experiments)")
    print("═"*(cw+66))
    print(f"  {'Rank':<5} {'Model Tag':<{cw}} {'Avg F1':>7} {'Avg Acc':>8} "
          f"{'Avg P':>7} {'Avg R':>7}  Experiments")
    print("  "+"─"*(cw+62))
    for rank, s in enumerate(srt, 1):
        exp_str = ", ".join(sorted(s["experiments"]))
        marker  = " ← BEST" if rank == 1 else ""
        print(f"  {rank:<5} {s['model_tag']:<{cw}} "
              f"{s['avg_f1']:>7.4f} {s['avg_acc']:>8.4f} "
              f"{s['avg_p']:>7.4f} {s['avg_r']:>7.4f}  {exp_str}{marker}")

    # ── E1-Only Fair Comparison ────────────────────────────────────────────
    e1_models = [(s["model_tag"], s["per_exp"]["E1"])
                 for s in all_scores if "E1" in s.get("per_exp", {})]
    if e1_models:
        e1_srt = sorted(e1_models, key=lambda x: x[1]["avg_f1"], reverse=True)
        print()
        print("═"*(cw+66))
        print("  E1-ONLY FAIR COMPARISON — Experiment 1: Cam A Train → Cam A Test")
        print("  (All models evaluated on same experiment — direct head-to-head)")
        print("═"*(cw+66))
        print(f"  {'Rank':<5} {'Model Tag':<{cw}} {'E1 F1':>7} {'E1 Acc':>8} "
              f"{'E1 P':>7} {'E1 R':>7}")
        print("  "+"─"*(cw+42))
        for rank, (model_tag, ep) in enumerate(e1_srt, 1):
            marker = " ← BEST (E1)" if rank == 1 else ""
            print(f"  {rank:<5} {model_tag:<{cw}} "
                  f"{ep['avg_f1']:>7.4f} {ep['avg_acc']:>8.4f} "
                  f"{ep['avg_p']:>7.4f} {ep['avg_r']:>7.4f}{marker}")

    # ── Table 3: 2-class per experiment ────────────────────────────────────
    print()
    print("═"*(cw+76))
    print("  TABLE 3: 2-Class Results [Normal vs Tampered]")
    print("  (4-class predictions grouped post-hoc)")
    print("  Cols: TN, FP, FN, TP  |  TPR, FPR, Acc  |  hFAR = false alarms/hr normal footage")
    print("═"*(cw+76))

    for exp_key in ["E1", "E2", "E3"]:
        exp_models = [(s["model_tag"], s["per_exp"][exp_key])
                      for s in all_scores if exp_key in s.get("per_exp", {})]
        if not exp_models:
            continue
        e2_note = "  [Covered+Defocused only; Moved excluded]" if exp_key == "E2" else ""
        print(f"\n  Experiment {exp_key[-1]} — {EXP_NAMES[exp_key]}{e2_note}")
        hdr = (f"  {'Model Tag':<{cw}}  {'TN':>10}  {'FP':>10}  {'FN':>10}  {'TP':>10}  "
               f"{'TPR':>7}  {'FPR':>7}  {'Acc':>7}  {'hFAR':>8}")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        # Sort by TPR descending within experiment
        exp_models_sorted = sorted(exp_models, key=lambda x: x[1]["binary_tpr"], reverse=True)
        for model_tag, ep in exp_models_sorted:
            hfar = ep["binary_hfar"]
            hfar_s = f"{hfar:>8.2f}" if hfar != float("inf") else "     inf"
            print(f"  {model_tag:<{cw}}  {ep['binary_tn']:>10,.0f}  {ep['binary_fp']:>10,.0f}  "
                  f"{ep['binary_fn']:>10,.0f}  {ep['binary_tp']:>10,.0f}  "
                  f"{ep['binary_tpr']:>7.4f}  {ep['binary_fpr']:>7.4f}  "
                  f"{ep['binary_acc']:>7.4f}  {hfar_s}")

    # ── Table 4: 4-class per-class TPR/FPR/Acc per experiment ──────────────
    print()
    print("═"*(cw+80))
    print("  TABLE 4: 4-Class Results [Per-Class TPR / FPR / Acc]")
    print("  (smoothed predictions, averaged across all test days per experiment)")
    print("  E2: Normal and Moved excluded (shown as  -- ); only Covered & Defocused evaluated")
    print("═"*(cw+80))

    for exp_key in ["E1", "E2", "E3"]:
        exp_models = [(s["model_tag"], s["per_exp"][exp_key])
                      for s in all_scores if exp_key in s.get("per_exp", {})]
        if not exp_models:
            continue
        is_e2 = (exp_key == "E2")
        e2_note = "  [Covered+Defocused only]" if is_e2 else ""
        print(f"\n  Experiment {exp_key[-1]} — {EXP_NAMES[exp_key]}{e2_note}")

        # Header
        def _ch(label, w=6): return f"{label:>{w}}"
        hdr = f"  {'Model Tag':<{cw}}"
        for cls in ["Normal", "Covered", "Defocused", "Moved"]:
            short = cls[:1]  # N, C, D, M
            hdr += f"  {short+'-TPR':>7} {short+'-FPR':>7} {short+'-Acc':>7}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        # Sort by Covered-TPR then Defocused-TPR
        exp_models_sorted = sorted(
            exp_models,
            key=lambda x: (x[1]["tpr_per"][1] + x[1]["tpr_per"][2]),
            reverse=True
        )
        for model_tag, ep in exp_models_sorted:
            row = f"  {model_tag:<{cw}}"
            for cls_id in range(4):
                if is_e2 and cls_id in (0, 3):
                    row += f"  {'--':>7} {'--':>7} {'--':>7}"
                else:
                    row += (f"  {ep['tpr_per'][cls_id]:>7.3f} "
                            f"{ep['fpr_per'][cls_id]:>7.3f} "
                            f"{ep['acc_per'][cls_id]:>7.3f}")
            print(row)

    print()
    # ── Best model callout ──
    best4 = srt[0]
    srt2  = sorted(all_scores, key=lambda x: x["avg_btpr"], reverse=True)
    best2 = srt2[0]
    print("═"*(cw+66))
    print(f"★  Best overall 4-class : {best4['model_tag']}  (Avg F1={best4['avg_f1']:.4f})")
    print(f"★  Best overall 2-class : {best2['model_tag']}  "
          f"(Avg TPR={best2['avg_btpr']:.4f}, hFAR={best2['avg_bhfar']:.2f}/hr)")
    print("═"*(cw+66))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    eval_root   = os.path.join(script_dir, "Evaluation Results")
    review_root = os.path.join(eval_root,  "Review Evaluation")
    os.makedirs(review_root, exist_ok=True)

    print("UHCTD Evaluation Review Script")
    print("="*60)
    print(f"Scanning : {eval_root}")
    print(f"Output   : {review_root}\n")

    model_map = discover_experiment_folders(eval_root)
    if not model_map:
        print("No experiment folders found!"); sys.exit(1)

    print(f"Found {len(model_map)} model tags across "
          f"{sum(len(v) for v in model_map.values())} folders:")
    for tag, exps in sorted(model_map.items()):
        print(f"  {tag}  [{', '.join(e for e,_ in sorted(exps))}]")
    print()

    all_scores = []

    for model_tag, exp_list in sorted(model_map.items()):
        review_dir = os.path.join(review_root, f"{model_tag}-Review")
        os.makedirs(review_dir, exist_ok=True)

        f1v=[]; accv=[]; pv=[]; rv=[]; tprv=[]; fprv=[]; accpv=[]
        btprv=[]; bfprv=[]; baccv=[]; bhfarv=[]
        experiments_seen = set()
        per_exp = {}   # per-experiment averages for Table 3 / Table 4

        for exp_key, folder in sorted(exp_list):
            is_e2 = (exp_key == "E2")
            print(f"Processing {exp_key}_{model_tag} …")
            day_results = []

            # Per-experiment accumulators
            exp_f1v=[]; exp_accv=[]; exp_pv=[]; exp_rv=[]
            exp_tprv=[]; exp_fprv=[]; exp_accpv=[]
            exp_btpv=[]; exp_bfpv=[]; exp_bacv=[]; exp_bhv=[]
            exp_btp=[]; exp_btn=[]; exp_bfp=[]; exp_bfn=[]

            for day in DAYS:
                csv_path = find_csv_for_day(folder, exp_key, model_tag, day)
                if not csv_path or not os.path.exists(csv_path):
                    print(f"  [{day}] not found — skip"); continue
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"  [{day}] read error: {e}"); continue

                yt   = df["true_label"].values.astype(int)
                yp_r = df["raw_prediction"].values.astype(int)
                yp_s = df["smoothed_prediction"].values.astype(int)

                # 4-class metrics
                eval_labels = [1, 2] if is_e2 else [0, 1, 2, 3]
                raw_m = compute_metrics(yt, yp_r, labels=eval_labels)
                sm_m  = compute_metrics(yt, yp_s, labels=eval_labels)

                # 2-class binary (Normal vs Tampered)
                b_raw = compute_binary_metrics(yt, yp_r, fps=FPS_CAM_A, exclude_moved=is_e2)
                b_sm  = compute_binary_metrics(yt, yp_s, fps=FPS_CAM_A, exclude_moved=is_e2)

                # Ground truth stats
                unique, counts = np.unique(yt, return_counts=True)
                gt_stats = {"total": len(yt)}
                for lbl, cnt in zip(unique, counts): gt_stats[int(lbl)] = int(cnt)
                for i in range(4): gt_stats.setdefault(i, 0)

                day_results.append(dict(
                    day=day, raw_m=raw_m, smoothed_m=sm_m,
                    binary_raw=b_raw, binary_sm=b_sm,
                    gt_stats=gt_stats, fps=FPS_CAM_A, n_windows=len(yt)//32
                ))

                n_eval = sm_m.get("n_filtered", len(yt))
                eval_note = f" [2cls:{n_eval:,}]" if is_e2 else ""
                print(f"  [{day}]{eval_note} "
                      f"4cls F1={sm_m['macro_f1']:.3f} Acc={sm_m['accuracy']:.3f} | "
                      f"2cls TPR={b_sm['tpr']:.3f} FPR={b_sm['fpr']:.3f} hFAR={b_sm['hfar']:.1f}")

                # Overall accumulators
                f1v.append(sm_m["macro_f1"]); accv.append(sm_m["accuracy"])
                pv.append(sm_m["macro_precision"]); rv.append(sm_m["macro_recall"])
                tprv.append(sm_m["tpr_per"]); fprv.append(sm_m["fpr_per"])
                accpv.append(sm_m["acc_per"])
                btprv.append(b_sm["tpr"]); bfprv.append(b_sm["fpr"])
                baccv.append(b_sm["acc"])
                if b_sm["hfar"] != float("inf"): bhfarv.append(b_sm["hfar"])

                # Per-experiment accumulators
                exp_f1v.append(sm_m["macro_f1"]); exp_accv.append(sm_m["accuracy"])
                exp_pv.append(sm_m["macro_precision"]); exp_rv.append(sm_m["macro_recall"])
                exp_tprv.append(sm_m["tpr_per"]); exp_fprv.append(sm_m["fpr_per"])
                exp_accpv.append(sm_m["acc_per"])
                exp_btpv.append(b_sm["tpr"]); exp_bfpv.append(b_sm["fpr"])
                exp_bacv.append(b_sm["acc"])
                if b_sm["hfar"] != float("inf"): exp_bhv.append(b_sm["hfar"])
                exp_btp.append(b_sm["tp"]); exp_btn.append(b_sm["tn"])
                exp_bfp.append(b_sm["fp"]); exp_bfn.append(b_sm["fn"])

            if not day_results:
                print(f"  No data for {exp_key}_{model_tag} — skipping"); continue
            experiments_seen.add(exp_key)

            # Store per-experiment summary for Table 3 / Table 4
            if exp_tprv:
                per_exp[exp_key] = dict(
                    avg_f1=float(np.mean(exp_f1v)), avg_acc=float(np.mean(exp_accv)),
                    avg_p=float(np.mean(exp_pv)),   avg_r=float(np.mean(exp_rv)),
                    tpr_per=np.mean(exp_tprv, axis=0).tolist(),
                    fpr_per=np.mean(exp_fprv, axis=0).tolist(),
                    acc_per=np.mean(exp_accpv, axis=0).tolist(),
                    binary_tpr=float(np.mean(exp_btpv)),
                    binary_fpr=float(np.mean(exp_bfpv)),
                    binary_acc=float(np.mean(exp_bacv)),
                    binary_hfar=float(np.mean(exp_bhv)) if exp_bhv else float("inf"),
                    # Summed TP/TN/FP/FN across all days for Table 3 counts
                    binary_tp=float(np.sum(exp_btp)),
                    binary_tn=float(np.sum(exp_btn)),
                    binary_fp=float(np.sum(exp_bfp)),
                    binary_fn=float(np.sum(exp_bfn)),
                )

            txt = build_report(exp_key, model_tag, day_results)
            txt_path = os.path.join(review_dir, f"{exp_key}_{model_tag}-Review.txt")
            with open(txt_path, "w", encoding="utf-8") as f: f.write(txt)
            print(f"  → {txt_path}")

        if f1v:
            all_scores.append(dict(
                model_tag=model_tag, experiments=experiments_seen,
                avg_f1=float(np.mean(f1v)), avg_acc=float(np.mean(accv)),
                avg_p=float(np.mean(pv)), avg_r=float(np.mean(rv)),
                per_cls_tpr=np.mean(tprv, axis=0).tolist(),
                per_cls_fpr=np.mean(fprv, axis=0).tolist(),
                per_cls_acc=np.mean(accpv, axis=0).tolist(),
                avg_btpr=float(np.mean(btprv)) if btprv else 0.,
                avg_bfpr=float(np.mean(bfprv)) if bfprv else 0.,
                avg_bacc=float(np.mean(baccv)) if baccv else 0.,
                avg_bhfar=float(np.mean(bhfarv)) if bhfarv else float("inf"),
                per_exp=per_exp,   # per-experiment breakdown for Table 3 / Table 4
            ))
        print()

    if all_scores:
        print_ranking(all_scores)
    else:
        print("No scores computed.")

    print("Review complete!")
    print(f"All TXT files written to: {review_root}")


if __name__ == "__main__":
    main()
