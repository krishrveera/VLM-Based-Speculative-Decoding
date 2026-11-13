#!/usr/bin/env python3
"""Create an instrumented COPY of MMSpec's MSD evaluator. The original files are never modified.

Writes:
  evaluation/eval_msd_e2e_mmspec.py          (copy of eval_msd_mmspec.py + 3 extra logged fields)
  scripts/<Model>/eval_msd_e2e_mmspec.sh     (copy of eval_msd_mmspec.sh pointing at it)

One run then gives BOTH timings and an unbiased MAT:
  wall_time        unchanged: the original boundary (embedding prep NOT timed)  -> reproduce the paper
  embed_prep_time  NEW: wall time of model.get_inputs_embeds(...): copies input_ids / pixel_values to the GPU,
                   token-embedding lookup, the vision encoder (model.visual, method/msd/utils.py:561) and
                   merging image features into the sequence. All of this happens inside generate() for the
                   baseline and ViSpec. CPU image preprocessing (processor resize/patching in build_prompt)
                   is outside every method's timer, the baseline's included, so it is consistently excluded.
                   -> end-to-end time = wall_time + embed_prep_time, comparable to the baseline's boundary
  accept_sum / accept_steps
                   NEW: accepted tokens and verify steps of THIS turn only (counter after − before).
                   Valid because model.acclen / model.accnum only ever increase (+= in
                   method/msd/ea_model.py:592-593 and 662-663) and are reset once, before the loop
                   (eval_msd_mmspec.py:175). That is also why the original acceptance_length is a running
                   average over the whole run. validate_msd_e2e.py check 5 confirms this on real output.

Usage (from the MMSpec repo root):
    python tools/make_msd_e2e.py                 # Qwen2.5-VL-7B
    python tools/make_msd_e2e.py LLaVA-1.5-7B
"""
import os
import sys

ROOT = os.getcwd()
MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-VL-7B"
SRC = os.path.join(ROOT, "evaluation", "eval_msd_mmspec.py")
DST = os.path.join(ROOT, "evaluation", "eval_msd_e2e_mmspec.py")
SH_SRC = os.path.join(ROOT, "scripts", MODEL_DIR, "eval_msd_mmspec.sh")
SH_DST = os.path.join(ROOT, "scripts", MODEL_DIR, "eval_msd_e2e_mmspec.sh")

I16, I12 = " " * 16, " " * 12
EDITS = [
    # 1) per-turn lists for the new fields
    (I12 + "turn_target_times = []\n" + I12 + "conversation_history = []\n",
     I12 + "turn_target_times = []\n" + I12 + "conversation_history = []\n"
     + I12 + "turn_embed_prep_times = []\n" + I12 + "turn_accept_sums = []\n" + I12 + "turn_accept_steps = []\n"),
    # 2) start a clock before the vision encoder runs
    (I16 + "# Get input embeds\n",
     I16 + "# Get input embeds  [e2e: timed separately]\n"
     + I16 + "torch.cuda.synchronize()\n" + I16 + "_enc_start = time.time()\n"),
    # 3) stop it where the original timer starts; snapshot the acceptance counters
    (I16 + "torch.cuda.synchronize()\n" + I16 + "start_time = time.time()\n",
     I16 + "torch.cuda.synchronize()\n" + I16 + "_embed_prep_time = time.time() - _enc_start\n"
     # copy VALUES: model.acclen becomes a GPU tensor updated IN PLACE (self.acclen += accept_length),
     # so a plain reference would change with it and every per-turn difference would be 0
     + I16 + "_acc0, _num0 = float(model.acclen), int(model.accnum)\n"
     + I16 + "start_time = time.time()\n"),
    # 4) record this turn's values next to the original ones
    (I16 + "turn_target_times.append(target_time)\n",
     I16 + "turn_target_times.append(target_time)\n"
     + I16 + "turn_embed_prep_times.append(_embed_prep_time)\n"
     + I16 + "turn_accept_sums.append(float(model.acclen) - _acc0)\n"
     + I16 + "turn_accept_steps.append(int(model.accnum - _num0))\n"),
    # 5) save them
    (I16 + "\"target_time\": turn_target_times,\n" + I12 + "})",
     I16 + "\"target_time\": turn_target_times,\n"
     + I16 + "\"embed_prep_time\": turn_embed_prep_times,\n"
     + I16 + "\"accept_sum\": turn_accept_sums,\n"
     + I16 + "\"accept_steps\": turn_accept_steps,\n" + I12 + "})"),
]


def main():
    src = open(SRC).read()
    out = src
    for i, (old, new) in enumerate(EDITS, 1):
        n = out.count(old)
        if n != 1:
            sys.exit(f"edit {i}: expected the anchor once in {SRC}, found {n}. "
                     "The repo version differs from commit 2904868; check before using.")
        out = out.replace(old, new)
    out = out.replace('Method: MSD', 'Method: MSD (e2e-instrumented copy)', 1)

    sh = open(SH_SRC).read()
    for old, new in [("evaluation.eval_msd_mmspec", "evaluation.eval_msd_e2e_mmspec"),
                     ("msd-temperature-", "msd_e2e-temperature-")]:
        if sh.count(old) != 1:
            sys.exit(f"{SH_SRC}: expected '{old}' once, found {sh.count(old)}. Nothing written.")
        sh = sh.replace(old, new)

    # every replacement checked above; only now write both files (no half-generated setup on failure)
    open(DST, "w").write(out)
    open(SH_DST, "w").write(sh)
    print("wrote", DST)
    print("wrote", SH_DST)
    print("run:  bash tools/run_mmspec.sh testmini baseline msd_e2e")


if __name__ == "__main__":
    main()
