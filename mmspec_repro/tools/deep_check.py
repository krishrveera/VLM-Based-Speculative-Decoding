import json, sys, statistics as st, collections, os
D = sys.argv[1]
M = ["baseline_greedy","baseline","eagle","eagle2","eagle3","vispec","msd","msd_e2e","medusa","pld","lookahead","recycling","sam"]
M = [m for m in M if os.path.exists(f"{D}/{m}-temperature-0.jsonl")]
def load(m): return {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{D}/{m}-temperature-0.jsonl") if l.strip()}
R = {m: load(m) for m in M}
Q = sorted(R["baseline_greedy"])
T = lambda m,q: "\n<T>\n".join(R[m][q]["choices"][0]["turns"])
N = lambda m,q: sum(R[m][q]["choices"][0]["new_tokens"])
print(f"== exact-match matrix (% of {len(Q)}) ==")
short = {"msd_e2e":"msd","medusa":"med","baseline_greedy":"greedy","baseline":"pen","eagle":"e1","eagle2":"e2","eagle3":"e3","vispec":"visp","msd":"msd","pld":"pld","lookahead":"look","recycling":"recy","sam":"sam"}
print("        " + " ".join(f"{short[m]:>6}" for m in M))
for a in M:
    print(f"{short[a]:>7} " + " ".join(f"{100*sum(T(a,q)==T(b,q) for q in Q)/len(Q):6.0f}" for b in M))
def rel(a, b, q):
    x, y = T(a,q), T(b,q)
    if x == y: return "same"
    if y.startswith(x): return "a_prefix_of_b"     # a stopped early
    if x.startswith(y): return "b_prefix_of_a"
    return "diverge"
print("\n== vs baseline_greedy: relation of each method's answer ==")
for m in M[1:]:
    c = collections.Counter(rel(m,"baseline_greedy",q) for q in Q)
    first=[]
    for q in Q:
        x,y=T(m,q),T("baseline_greedy",q)
        if x!=y and not (x.startswith(y) or y.startswith(x)):
            i=next(i for i,(u,v) in enumerate(zip(x,y)) if u!=v); first.append(i/max(1,len(y)))
    print(f"{m:10} {dict(c)}" + (f"  diverge at median {st.median(first):.0%} of answer" if first else ""))
for m in [x for x in ["msd","msd_e2e","sam"] if x in M]:
    print(f"\n== {m}: where it is shorter than baseline_greedy ==")
    rows=[(q,N(m,q),N("baseline_greedy",q),R[m][q]["topic"],rel(m,"baseline_greedy",q)) for q in Q if N(m,q) < 0.8*N("baseline_greedy",q)]
    print(f"{len(rows)} samples with <80% of greedy's tokens; by task:", dict(collections.Counter(r[3] for r in rows)))
    print("relation for those:", dict(collections.Counter(r[4] for r in rows)))
    for q,a,b,t,r in rows[:6]: print(f"  {q} {t[:18]:18} {m}={a:4} greedy={b:4} {r}")
    # endings
    ends=collections.Counter((T(m,q).rstrip()[-1:] or "∅") for q,_,_,_,_ in rows)
    print("last char of the shorter answers:", dict(ends.most_common(8)))
print("\n== max new tokens per method (cap check) ==")
for m in M:
    toks=[t for q in Q for t in R[m][q]["choices"][0]["new_tokens"]]
    print(f"{m:16} max {max(toks):5}  turns>=1000: {sum(t>=1000 for t in toks):3}  mean {st.mean(toks):6.1f}")
