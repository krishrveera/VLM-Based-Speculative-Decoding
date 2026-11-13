#!/usr/bin/env python3
"""Summarize MMSpec result files into a Table-3-style comparison.

MMSpec's repo writes one JSONL per method (results/<Model>/mmspec_<split>/<method>-temperature-<T>.jsonl)
but ships no script that turns them into MAT / speedup. This does, and adds a lossless check.

Usage:
    python summarize_mmspec.py results/Qwen2.5-VL-7B/mmspec_test \
        [--baseline baseline-temperature-0.jsonl] [--model qwen] [--out summary]

Writes <out>.md (readable table) and <out>.csv (one row per method x topic).

Definitions (fixed for the team; see the runbook):
  MAT          mean accepted draft tokens per verify step, bonus token excluded
               (EAGLE-style evaluate_posterior accept_length, as MMSpec logs it), step-weighted.
               Original MSD output stores a running average instead; see MAT kind in the output.
  aggregation  several speedup averages are reported side by side (pooled tok/s, pooled time,
               mean and median of per-sample ratios, end-to-end) because the paper does not say which it used.
  tau          MAT + 1 = tokens committed per target forward (vLLM-style "acceptance length").
  speedup_tps  (method tokens / method seconds) / (baseline tokens / baseline seconds), pooled per topic.
               Spec-Bench convention, which MMSpec says it follows.
  speedup_time sum(baseline seconds) / sum(method seconds) over the same samples.
               Equals speedup_tps when outputs are identical (greedy, lossless).
  exact_match  share of samples whose full output text equals the baseline's. Should be ~100% at T=0.
"""
import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

# Repo "topic" field -> column names in the paper's Table 3
TOPIC_COLS = {
    "general vqa": "GQA",
    "text vqa": "TVQA",
    "image captioning": "IC",
    "chart understanding": "CQA",
    "complex reasoning pro": "CR",
    "multi-turn conversation": "MTC",
}
COL_ORDER = ["GQA", "TVQA", "IC", "CQA", "CR", "MTC", "Overall"]

# MMSpec arXiv 2603.14989v1, Table 3, Qwen2.5-VL-7B, T=0: (MAT, speedup) per column.
PAPER_QWEN = {
    "eagle":     [(2.41, 1.81), (2.04, 1.28), (2.46, 2.59), (2.24, 1.93), (2.46, 2.22), (2.33, 2.14), (2.36, 2.11)],
    "eagle2":    [(1.79, 1.73), (1.52, 1.17), (1.70, 2.32), (1.87, 1.95), (1.86, 2.56), (1.78, 1.92), (1.78, 2.02)],
    "eagle3":    [(0.43, 1.02), (0.22, 0.83), (0.38, 1.17), (0.22, 1.09), (0.16, 0.85), (0.25, 0.97), (0.24, 0.96)],
    "medusa":    [(0.82, 1.29), (0.66, 1.01), (0.72, 1.55), (0.92, 1.59), (0.82, 1.99), (0.82, 1.38), (0.80, 1.49)],
    "msd":       [(2.32, 2.27), (2.37, 1.80), (2.35, 2.03), (2.67, 1.96), (2.42, 4.06), (2.51, 2.78), (2.57, 2.58)],
    "vispec":    [(2.01, 1.67), (1.26, 1.12), (1.75, 2.07), (1.57, 1.72), (1.28, 1.52), (1.15, 1.42), (1.29, 1.51)],
    "lookahead": [(0.20, 1.06), (0.07, 0.77), (0.11, 1.08), (0.43, 1.34), (0.65, 1.47), (0.30, 0.95), (0.33, 1.07)],
    "recycling": [(0.05, 1.02), (0.04, 0.80), (0.05, 1.04), (0.17, 0.87), (0.13, 1.08), (0.12, 1.09), (0.11, 1.04)],
    "pld":       [(0.03, 1.01), (0.00, 0.78), (0.00, 1.00), (0.29, 1.04), (0.26, 1.16), (0.20, 1.06), (0.17, 1.05)],
    "sam":       [(0.11, 1.33), (0.10, 1.23), (0.16, 1.48), (0.51, 2.04), (0.19, 6.53), (0.27, 2.04), (0.23, 2.17)],
}


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            c = r["choices"][0]
            acc = c.get("acceptance_length") or []
            # Three formats exist in MMSpec output:
            #  per-step lists (EAGLE/ViSpec-style)       -> exact, step-weighted
            #  accept_sum/accept_steps (our MSD e2e copy) -> exact, step-weighted
            #  one float per turn (original MSD)          -> running average over the whole run; unreliable per task
            if c.get("accept_steps") is not None:
                a_sum, a_steps, mat_kind = float(sum(c["accept_sum"])), int(sum(c["accept_steps"])), "exact"
            elif any(isinstance(t, list) for t in acc):
                flat = [x for t in acc if isinstance(t, list) for x in t if x is not None]
                a_sum, a_steps, mat_kind = float(sum(flat)), len(flat), "exact"
            elif acc:
                a_sum, a_steps, mat_kind = float(acc[-1]), 1, "running_avg"
            else:
                a_sum, a_steps, mat_kind = 0.0, 0, "none"
            wall = float(sum(c.get("wall_time") or [0.0]))
            enc = c.get("embed_prep_time")
            rows[r["question_id"]] = {
                "topic": TOPIC_COLS.get(r.get("topic", ""), r.get("topic", "unknown")),
                "time": wall,
                "time_e2e": wall + float(sum(enc)) if enc is not None else None,
                "tokens": int(sum(c.get("new_tokens") or [0])),
                "turn_tokens": list(c.get("new_tokens") or []),
                "text": "\n<turn>\n".join(c.get("turns") or []),
                "a_sum": a_sum, "a_steps": a_steps, "mat_kind": mat_kind,
                "tstamp": r.get("tstamp", 0),
                "_run_avg": [float(x) for x in acc] if acc and not any(isinstance(t, list) for t in acc) else None,
                "_steps": list(c["accept_steps"]) if c.get("accept_steps") is not None else None,
            }
    _recover_msd_acceptance(rows)
    return rows


def _recover_msd_acceptance(rows):
    """MSD runs that log accept_steps also keep the original running average r_k = cumAcc_k / cumSteps_k.
    Replaying turns in run order gives exact per-turn accepted tokens: cumAcc_k - cumAcc_(k-1).
    This is independent of accept_sum, so it also repairs msd_e2e files written before the
    in-place-tensor fix (accept_sum stuck at 0). Valid only for a run that was not resumed mid-way."""
    if not rows or any(r["_steps"] is None or r["_run_avg"] is None for r in rows.values()):
        return
    cum_steps, prev = 0, 0.0
    for q in sorted(rows, key=lambda q: rows[q]["tstamp"]):
        r = rows[q]
        acc = 0.0
        for avg, st in zip(r["_run_avg"], r["_steps"]):
            cum_steps += st
            cum = avg * cum_steps
            acc += cum - prev
            prev = cum
        r["a_sum"], r["a_steps"], r["mat_kind"] = acc, int(sum(r["_steps"])), "exact"


def method_name(path):
    return os.path.basename(path).split("-temperature-")[0]


def summarize(base, meth):
    groups = defaultdict(list)
    for qid, m in meth.items():
        if qid in base:
            groups[m["topic"]].append(qid)
    groups["Overall"] = [q for col in list(groups) for q in groups[col]]
    out = {}
    for col, qids in groups.items():
        if not qids:
            continue
        bt = sum(base[q]["time"] for q in qids)
        mt = sum(meth[q]["time"] for q in qids)
        btok = sum(base[q]["tokens"] for q in qids)
        mtok = sum(meth[q]["tokens"] for q in qids)
        kinds = {meth[q]["mat_kind"] for q in qids}
        if kinds == {"exact"}:
            steps = sum(meth[q]["a_steps"] for q in qids)
            mat = sum(meth[q]["a_sum"] for q in qids) / steps if steps else float("nan")
        elif "running_avg" in kinds:
            # original MSD: the value saved with the LAST-run sample is the cumulative average up to it.
            # Correct for "Overall" (last sample overall); NOT a per-task average for other columns.
            last = max(qids, key=lambda q: meth[q]["tstamp"])
            mat = meth[last]["a_sum"]
        else:
            mat = float("nan")
        per_ratio = [base[q]["time"] / meth[q]["time"] for q in qids if meth[q]["time"] > 0]
        e2e = [meth[q]["time_e2e"] for q in qids]
        out[col] = {
            "n": len(qids),
            "MAT": mat,
            "MAT_kind": "/".join(sorted(kinds)),
            "speedup_tps": (mtok / mt) / (btok / bt) if mt and bt and btok else float("nan"),
            "speedup_time": bt / mt if mt else float("nan"),
            "speedup_mean_per_sample": statistics.mean(per_ratio) if per_ratio else float("nan"),
            "speedup_median_per_sample": statistics.median(per_ratio) if per_ratio else float("nan"),
            "speedup_e2e": bt / sum(e2e) if all(x is not None for x in e2e) and sum(e2e) else float("nan"),
            "lat_p50_s": pct([meth[q]["time"] for q in qids], 50),
            "lat_p95_s": pct([meth[q]["time"] for q in qids], 95),
            "base_lat_p50_s": pct([base[q]["time"] for q in qids], 50),
            "base_lat_p95_s": pct([base[q]["time"] for q in qids], 95),
            "tokens_ratio": mtok / btok if btok else float("nan"),
            # "turns" = decoded generated text (process_output: tokens after the prompt, special tokens and
            # outer whitespace stripped). Token IDs are not saved, so we also compare per-turn token counts.
            "exact_match": sum(base[q]["text"] == meth[q]["text"] for q in qids) / len(qids),
            "same_length": sum(base[q]["turn_tokens"] == meth[q]["turn_tokens"] for q in qids) / len(qids),
            "base_tok_s": btok / bt if bt else float("nan"),
        }
    return out


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def fmt(x, nd=2):
    return "—" if x != x else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--baseline", default=None, help="baseline file name (default: baseline-temperature-*.jsonl)")
    ap.add_argument("--model", default="qwen", choices=["qwen", "none"], help="paper reference to diff against")
    ap.add_argument("--out", default=None)
    ap.add_argument("--expected", type=int, default=None,
                    help="samples per complete run (default: 60 for *_testmini, 600 for *_test)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="also list failed/incomplete methods in the table (marked); default: excluded")
    a = ap.parse_args()

    bpath = os.path.join(a.results_dir, a.baseline) if a.baseline else \
        (sorted(glob.glob(os.path.join(a.results_dir, "baseline-temperature-*.jsonl"))) or [None])[0]
    if not bpath or not os.path.exists(bpath):
        raise SystemExit("No baseline file found; run eval_baseline_mmspec.sh first.")
    base = load(bpath)
    ref = PAPER_QWEN if a.model == "qwen" else {}

    d = os.path.basename(os.path.normpath(a.results_dir))
    expected = a.expected or (60 if d.endswith("_testmini") else 600 if d.endswith("_test") else len(base))
    status = {}
    spath = os.path.join(a.results_dir, "logs", "status.tsv")
    if os.path.exists(spath):
        for ln in open(spath):
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 2:
                status[parts[0]] = parts[1]          # last attempt wins

    def check(name, rows):
        """Return a problem string, or '' if the run is complete and did not fail."""
        probs = []
        if status.get(name, "0") != "0":
            probs.append(f"last run exited {status[name]}")
        if len(rows) < expected:
            probs.append(f"{len(rows)}/{expected} samples")
        return "; ".join(probs)

    lines = [f"# MMSpec summary — {a.results_dir}", ""]
    bprob = check("baseline", base)
    if bprob:
        lines += [f"> **WARNING: baseline incomplete ({bprob}).** Every speedup below uses only the samples the "
                  "baseline finished. Rerun the baseline before trusting any number.", ""]
    lines += [f"Baseline: `{os.path.basename(bpath)}` ({len(base)}/{expected} samples). "
              "Cells: MAT / speedup_tps (paper speedup in brackets). EM = exact text match vs baseline; "
              "Len = same token count per turn.", ""]
    excluded = []
    lines.append("| Method | " + " | ".join(COL_ORDER) + " | EM | Len | n |")
    lines.append("|" + " --- |" * (len(COL_ORDER) + 4))
    csv_rows = []
    detail = ["", "## Overall: aggregation check", "",
              "Same runs, different ways to average. If they disagree a lot, say which one any claim uses.", "",
              "| Method | MAT (kind) | Pooled tok/s | Pooled time | Mean per-sample | Median per-sample "
              "| End-to-end | p50 / p95 latency (s) | Baseline p50 / p95 (s) | Tokens vs base |",
              "|" + " --- |" * 10]
    for p in sorted(glob.glob(os.path.join(a.results_dir, "*-temperature-*.jsonl"))):
        if p == bpath:
            continue
        name = method_name(p)
        mrows = load(p)
        prob = check(name, mrows)
        if prob:
            excluded.append(f"- `{name}`: {prob}")
            if not a.allow_incomplete:
                continue
            name = f"⚠ {name}"
        s = summarize(base, mrows)
        cells = []
        for i, col in enumerate(COL_ORDER):
            if col not in s:
                cells.append("—")
                continue
            r = s[col]
            paper = ref.get(name) or ref.get(name.replace("_e2e", ""))
            pref = f" [{paper[i][1]:.2f}×]" if paper else ""
            cells.append(f"{fmt(r['MAT'])} / {fmt(r['speedup_tps'])}×{pref}")
            csv_rows.append({"method": name, "topic": col, **{k: round(v, 4) if isinstance(v, float) else v
                                                                for k, v in r.items()},
                             "paper_MAT": paper[i][0] if paper else "", "paper_speedup": paper[i][1] if paper else ""})
        ov = s.get("Overall", {})
        lines.append(f"| {name} | " + " | ".join(cells) +
                     f" | {fmt(ov.get('exact_match', float('nan')) * 100, 0)}% "
                     f"| {fmt(ov.get('same_length', float('nan')) * 100, 0)}% | {ov.get('n', 0)} |")
        if ov:
            detail.append(
                f"| {name} | {fmt(ov['MAT'])} ({ov['MAT_kind']}) | {fmt(ov['speedup_tps'])}× | {fmt(ov['speedup_time'])}× "
                f"| {fmt(ov['speedup_mean_per_sample'])}× | {fmt(ov['speedup_median_per_sample'])}× "
                f"| {fmt(ov['speedup_e2e'])}{'×' if ov['speedup_e2e'] == ov['speedup_e2e'] else ''} "
                f"| {fmt(ov['lat_p50_s'])} / {fmt(ov['lat_p95_s'])} | {fmt(ov['base_lat_p50_s'])} / {fmt(ov['base_lat_p95_s'])} "
                f"| {fmt(ov['tokens_ratio'])} |")

    detail += ["",
               "MAT kind: `exact` = accepted tokens ÷ verify steps; `running_avg` = original MSD output, a cumulative "
               "average over the whole run, so per-task MAT cells for it are not task averages (use the `msd_e2e` run).",
               "End-to-end: only for runs that log `embed_prep_time` (`msd_e2e`); adds embedding-prep time (GPU copies, token embedding, vision encoder) that the original "
               "MSD timer leaves out. Tokens vs base ≠ 1.00 means outputs differ in length: check before trusting speedup."]
    lines += detail
    if excluded:
        lines += ["", "## Not in the table (failed or incomplete)" if not a.allow_incomplete
                  else "## Marked ⚠ (failed or incomplete)", ""] + excluded + \
                 ["", "Rerun these methods; MMSpec skips finished samples, so a rerun only fills the gaps."]

    out = a.out or os.path.join(a.results_dir, "summary")
    with open(out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    if csv_rows:
        with open(out + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print("\n".join(lines))
    print(f"\nWrote {out}.md and {out}.csv")


if __name__ == "__main__":
    main()
