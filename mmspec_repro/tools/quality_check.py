"""Flag degenerate output that a lossless method can never produce: doubled words ('are are'),
junk tokens, and non-Latin fragments in an English answer. Counts per method, vs the greedy baseline."""
import json, re, sys, collections, os
D = sys.argv[1]
M = ["baseline_greedy","baseline","eagle","eagle2","eagle3","vispec","msd","msd_e2e","medusa","pld","lookahead","recycling","sam"]
M = [m for m in M if os.path.exists(f"{D}/{m}-temperature-0.jsonl")]
SAFE = {"that","had","is","the","very","bye","no","yes","ha","so","many","far","long","more","again","really","what","on"}
dbl = re.compile(r"\b([A-Za-z]{2,})\s+\1\b", re.I)
junk = re.compile(r"addCriterion|[一-鿿぀-ヿ]{1,}")
print(f"{'method':16}{'doubled-word answers':>22}{'junk-token answers':>20}{'any':>8}")
for m in M:
    rows=[json.loads(l) for l in open(f"{D}/{m}-temperature-0.jsonl") if l.strip()]; n=len(rows)
    d=j=a=0
    for r in rows:
        t=" ".join(r["choices"][0]["turns"])
        hd=any(x.group(1).lower() not in SAFE for x in dbl.finditer(t)); hj=bool(junk.search(t))
        d+=hd; j+=hj; a+=(hd or hj)
    print(f"{m:16}{d:>15}/{n}{j:>15}/{n}{a:>5}/{n}")
