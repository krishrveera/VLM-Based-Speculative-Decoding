#!/usr/bin/env python3
"""Create a PURE-GREEDY copy of MMSpec's baseline. The original files are never modified.

Why: the original baseline calls HF model.generate(do_sample=False), which still applies the
model's generation_config. Qwen2.5-VL-7B-Instruct ships repetition_penalty=1.05, so the baseline
decodes with a repetition penalty while the speculative methods decode pure greedy. Their outputs
then diverge early (MMSpec testmini on Nov 5: 0-10% exact match for every method) and speedups
compare answers of different content and length.

Writes:
  evaluation/eval_baseline_greedy_mmspec.py       (copy + repetition_penalty=1.0 forced)
  scripts/<Model>/eval_baseline_greedy_mmspec.sh  (copy writing baseline_greedy-temperature-*.jsonl)

Usage (from the MMSpec repo root):  python tools/make_baseline_greedy.py [Qwen2.5-VL-7B]
Then summarize against it:  python tools/summarize_mmspec.py <dir> --baseline baseline_greedy-temperature-0.jsonl
"""
import os
import sys

ROOT = os.getcwd()
MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-VL-7B"
SRC = os.path.join(ROOT, "evaluation", "eval_baseline_mmspec.py")
DST = os.path.join(ROOT, "evaluation", "eval_baseline_greedy_mmspec.py")
SH_SRC = os.path.join(ROOT, "scripts", MODEL_DIR, "eval_baseline_mmspec.sh")
SH_DST = os.path.join(ROOT, "scripts", MODEL_DIR, "eval_baseline_greedy_mmspec.sh")

OLD = '        "max_new_tokens": max_steps,\n        "use_cache": True,\n    }\n'
NEW = ('        "max_new_tokens": max_steps,\n        "use_cache": True,\n'
       '        # [greedy copy] override generation_config defaults (Qwen2.5-VL: repetition_penalty=1.05)\n'
       '        "repetition_penalty": 1.0,\n    }\n')


def main():
    src = open(SRC).read()
    if src.count(OLD) != 1:
        sys.exit(f"{SRC}: anchor found {src.count(OLD)} times (expected 1). Nothing written.")
    out = src.replace(OLD, NEW)
    sh = open(SH_SRC).read()
    for old, new in [("evaluation.eval_baseline_mmspec", "evaluation.eval_baseline_greedy_mmspec"),
                     ("baseline-temperature-", "baseline_greedy-temperature-")]:
        if sh.count(old) != 1:
            sys.exit(f"{SH_SRC}: expected '{old}' once, found {sh.count(old)}. Nothing written.")
        sh = sh.replace(old, new)
    open(DST, "w").write(out)
    open(SH_DST, "w").write(sh)
    print("wrote", DST)
    print("wrote", SH_DST)


if __name__ == "__main__":
    main()
