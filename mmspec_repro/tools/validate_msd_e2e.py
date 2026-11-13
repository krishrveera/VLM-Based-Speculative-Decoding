#!/usr/bin/env python3
"""Check that the instrumented MSD copy (msd_e2e) measures what it claims, using REAL outputs.

Needs both runs in one results folder, same GPU, same split:
    bash tools/run_mmspec.sh testmini msd msd_e2e
    python tools/validate_msd_e2e.py results/Qwen2.5-VL-7B/mmspec_testmini

Checks (PASS / FAIL / WARN each):
  1 same samples          both runs cover the same question ids
  2 same outputs          identical generated text and per-turn token counts (instrumentation must not change decoding)
  3 same decode timing    median msd_e2e wall_time / msd wall_time within ±5% (the added clock is outside the timed region)
  4 embed-prep time logged    embed_prep_time > 0 for every turn
  5 acceptance identity   msd_e2e still writes the ORIGINAL running-average field; replaying our per-turn
                          accept_sum/accept_steps in run order must reproduce it exactly.
                          This proves the per-turn counts are complete and correctly separated.
  6 overall MAT agrees    exact overall MAT (msd_e2e) equals the last running average of the original msd run
Also prints how big embedding-prep time (incl. vision encoder) is relative to decode time, per task.
"""
import json
import os
import statistics
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
orig_p = os.path.join(d, "msd-temperature-0.jsonl")
e2e_p = os.path.join(d, "msd_e2e-temperature-0.jsonl")


def load(p):
    rows = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            rows[r["question_id"]] = r
    return rows


o, e = load(orig_p), load(e2e_p)
results = []


def report(name, ok, msg, warn=False):
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    results.append(tag)
    print(f"[{tag}] {name}: {msg}")


# 1
common = sorted(set(o) & set(e))
report("1 same samples", set(o) == set(e), f"{len(common)} shared; only-orig {len(set(o)-set(e))}, only-e2e {len(set(e)-set(o))}")

# 2
same_text = sum(o[q]["choices"][0]["turns"] == e[q]["choices"][0]["turns"] for q in common)
same_len = sum(o[q]["choices"][0]["new_tokens"] == e[q]["choices"][0]["new_tokens"] for q in common)
worst_share = min(same_text, same_len) / len(common) if common else 0
report("2 same outputs", worst_share == 1.0,
       f"text identical {same_text}/{len(common)}, token counts identical {same_len}/{len(common)}"
       + ("" if worst_share == 1.0 else " — inspect the differing samples before using either run"),
       warn=worst_share >= 0.95)   # rare BF16 nondeterminism can flip a token; more than 5% is a real problem

# 3
ratios = []
for q in common:
    for a, b in zip(o[q]["choices"][0]["wall_time"], e[q]["choices"][0]["wall_time"]):
        if a > 0:
            ratios.append(b / a)
med = statistics.median(ratios) if ratios else float("nan")
report("3 same decode timing", abs(med - 1) <= 0.05, f"median e2e/orig wall_time = {med:.3f} over {len(ratios)} turns",
       warn=abs(med - 1) <= 0.10)

# 4
enc = [t for q in common for t in e[q]["choices"][0].get("embed_prep_time", [])]
report("4 embed-prep time logged", bool(enc) and all(t > 0 for t in enc),
       f"{len(enc)} turns, min {min(enc) if enc else float('nan'):.4f}s, median {statistics.median(enc) if enc else float('nan'):.4f}s")

# 5  replay in run order (tstamp = when the sample finished; turns in order within a sample)
order = sorted(e, key=lambda q: e[q].get("tstamp", 0))
cs = cn = 0.0
worst = 0.0
checked = 0
for q in order:
    c = e[q]["choices"][0]
    for run_avg, s, n in zip(c.get("acceptance_length", []), c.get("accept_sum", []), c.get("accept_steps", [])):
        cs += s
        cn += n
        if cn > 0:
            worst = max(worst, abs(cs / cn - run_avg))
            checked += 1
report("5 acceptance identity", checked > 0 and worst < 1e-6,
       f"replayed {checked} turns, max |cumulative(ours) − original running avg| = {worst:.2e}"
       + ("" if worst < 1e-6 else
          ". If this msd_e2e run crashed and was resumed, the original counter restarted mid-run: "
          "delete its file and run it once, uninterrupted, before judging this check."))

# 6
exact_mat = cs / cn if cn else float("nan")
last_orig = o[max(o, key=lambda q: o[q].get("tstamp", 0))]["choices"][0]["acceptance_length"][-1]
report("6 overall MAT agrees", abs(exact_mat - last_orig) < 0.02,
       f"exact overall MAT (e2e) {exact_mat:.3f} vs last running avg (orig) {last_orig:.3f}")

# size of the effect, per task
print("\nEmbedding-prep time (incl. vision encoder) vs decode time (msd_e2e):")
by = {}
for q in common:
    c = e[q]["choices"][0]
    by.setdefault(e[q].get("topic", "?"), []).append((sum(c.get("embed_prep_time", [0])), sum(c["wall_time"])))
for t, xs in sorted(by.items()):
    enc_s, dec_s = sum(x for x, _ in xs), sum(y for _, y in xs)
    print(f"  {t:<26} embed-prep {enc_s:7.2f}s  decode {dec_s:8.2f}s  share of e2e {enc_s/(enc_s+dec_s)*100:5.1f}%")

print(f"\n{results.count('PASS')} pass, {results.count('WARN')} warn, {results.count('FAIL')} fail")
sys.exit(1 if "FAIL" in results else 0)
