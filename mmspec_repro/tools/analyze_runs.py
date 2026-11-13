"""Run analysis (any split): corruption categories, bootstrap CIs, paired comparisons (pooled tok/s vs baseline_greedy)."""
import json, re, sys, random, collections, os, statistics as st
D = sys.argv[1]
M = ["eagle","vispec","eagle2","medusa","msd_e2e","pld","lookahead","sam","recycling","eagle3","baseline"]
M = [m for m in M if os.path.exists(f"{D}/{m}-temperature-0.jsonl")]
def load(m): return {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{D}/{m}-temperature-0.jsonl") if l.strip()}
G = load("baseline_greedy"); R = {m: load(m) for m in M}; Q = sorted(G); n = len(Q)
SAFE = {"that","had","is","the","very","bye","no","yes","ha","so","many","far","long","more","again","really","what","on"}
dbl = re.compile(r"\b([A-Za-z]{2,})\s+\1\b", re.I); cjk = re.compile(r"[一-鿿぀-ヿ]+")
END = (".", "!", "?", '"', ")", "*", "`", "'", ":")
def classify(m, q):
    gt = " ".join(G[q]["choices"][0]["turns"]); tail = mid = False
    for t in R[m][q]["choices"][0]["turns"]:
        for x in re.finditer("addCriterion", t):
            if t[:x.start()].rstrip().endswith(END): tail = True
            else: mid = True
        if any(x.group(1).lower() not in SAFE for x in dbl.finditer(t)) and not any(x.group(0) in gt for x in dbl.finditer(t)): mid = True
        if any(c not in gt for c in cjk.findall(t)): mid = True
    return tail, mid
print(f"== answer problems (of {n}); greedy baseline for reference ==")
print(f"{'method':12}{'mid-answer corruption':>24}{'junk after finished answer':>30}")
for m in M:
    c = [classify(m, q) for q in Q]
    print(f"{m:12}{sum(x[1] for x in c):>17} ({100*sum(x[1] for x in c)/n:4.1f}%){sum(x[0] and not x[1] for x in c):>21} ({100*sum(x[0] and not x[1] for x in c)/n:4.1f}%)")
tok = lambda m, q: sum((R[m] if m in R else G)[q]["choices"][0]["new_tokens"])
def tim(m, q, e2e=False):
    c = (R[m] if m in R else G)[q]["choices"][0]
    return sum(c["wall_time"]) + (sum(c.get("embed_prep_time", [0])) if e2e else 0)
def spd(m, idx, e2e=False):
    bt = sum(tok("baseline_greedy", Q[i]) for i in idx) / sum(tim("baseline_greedy", Q[i]) for i in idx)
    return (sum(tok(m, Q[i]) for i in idx) / sum(tim(m, Q[i], e2e) for i in idx)) / bt
random.seed(0); B = 2000
boots = [[random.randrange(n) for _ in range(n)] for _ in range(B)]
full = list(range(n))
def ci(v): v = sorted(v); return v[int(.025*len(v))], v[int(.975*len(v))-1]
print(f"\n== pooled tok/s speedup vs baseline_greedy, 95% bootstrap CI ({B} resamples of questions) ==")
S = {}
for m in M + (["msd_e2e*"] if "msd_e2e" in M else []):
    e2e = m.endswith("*"); mm = m.rstrip("*")
    S[m] = [spd(mm, b, e2e) for b in boots]
    lo, hi = ci(S[m]); print(f"{m:12} {spd(mm, full, e2e):.2f}x  ({lo:.2f}-{hi:.2f})")
print("   msd_e2e* = MSD with vision-encoder time included in its timer")
print("\n== paired differences (same resampled questions for both): A minus B ==")
for a, b in [p for p in [("eagle","vispec"),("eagle","eagle2"),("vispec","eagle2"),("msd_e2e","eagle"),("msd_e2e","vispec"),("msd_e2e*","eagle"),("msd_e2e*","vispec"),("eagle","medusa")] if p[0].rstrip("*") in M and p[1].rstrip("*") in M]:
    d = [x - y for x, y in zip(S[a], S[b])]; lo, hi = ci(d)
    pt = spd(a.rstrip("*"), full, a.endswith("*")) - spd(b.rstrip("*"), full, b.endswith("*"))
    verdict = "A faster" if lo > 0 else ("B faster" if hi < 0 else "cannot tell apart")
    print(f"{a:9} vs {b:8} {pt:+.3f}  ({lo:+.3f} to {hi:+.3f})  {verdict}")
