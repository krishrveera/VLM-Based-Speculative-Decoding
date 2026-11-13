"""One-figure summary: where each method was claimed to be vs. where it lands when measured fairly.
Usage: python tools/make_overview_figure.py runs/l4_test_600 analysis/l4_test_600/summary_vs_greedy.csv reports/figures"""
import csv, json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

RUN, CSV, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
INK, INK2, INK3, GRID, SURF = "#111111", "#55544f", "#8a8984", "#e6e5e1", "#ffffff"
BLUE, RED, PAPER = "#2a78d6", "#d03b3b", "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 240,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK3})
NAME = {"eagle": "EAGLE-1", "eagle2": "EAGLE-2", "eagle3": "EAGLE-3", "vispec": "ViSpec", "msd_e2e": "MSD",
        "medusa": "Medusa", "pld": "PLD", "lookahead": "Lookahead", "sam": "SAM", "recycling": "Recycling"}
S = {(r["method"], r["topic"]): r for r in csv.DictReader(open(CSV))}
o = lambda m, k: float(S[(m, "Overall")][k])

def load(m): return {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{RUN}/{m}-temperature-0.jsonl") if l.strip()}
G = load("baseline_greedy"); Q = sorted(G); n = len(Q)
SAFE = {"that","had","is","the","very","bye","no","yes","ha","so","many","far","long","more","again","really","what","on"}
dbl = re.compile(r"\b([A-Za-z]{2,})\s+\1\b", re.I); cjk = re.compile(r"[一-鿿぀-ヿ]+")
END = (".", "!", "?", '"', ")", "*", "`", "'", ":")
def intact(m):
    R = load(m); bad = 0
    for q in Q:
        gt = " ".join(G[q]["choices"][0]["turns"]); mid = False
        for t in R[q]["choices"][0]["turns"]:
            mid |= any(not t[:x.start()].rstrip().endswith(END) for x in re.finditer("addCriterion", t))
            mid |= any(x.group(1).lower() not in SAFE for x in dbl.finditer(t)) and not any(x.group(0) in gt for x in dbl.finditer(t))
            mid |= any(c not in gt for c in cjk.findall(t))
        bad += mid
    return 100 * (1 - bad / n), R
data = {}
for m in NAME:
    y, R = intact(m)
    fair = None
    if m == "msd_e2e":
        bt = sum(sum(G[q]["choices"][0]["new_tokens"]) for q in Q) / sum(sum(G[q]["choices"][0]["wall_time"]) for q in Q)
        mt = sum(sum(R[q]["choices"][0]["new_tokens"]) for q in Q) / sum(sum(R[q]["choices"][0]["wall_time"]) + sum(R[q]["choices"][0]["embed_prep_time"]) for q in Q)
        fair = mt / bt
    data[m] = dict(paper=o(m, "paper_speedup"), ours=o(m, "speedup_tps"), fair=fair, y=y)

TYPE = {"eagle": "trained · text only", "eagle2": "trained · text only", "eagle3": "trained · text only",
        "medusa": "trained · text only", "vispec": "trained · sees image", "msd_e2e": "trained · sees image",
        "pld": "copy-based", "lookahead": "copy-based", "sam": "copy-based", "recycling": "copy-based"}
def verdict(m, d):
    gap = 100 * (1 - d["ours"] / d["paper"])
    if m == "msd_e2e": return ("Lead vanishes once the vision", "encoder is timed; corrupts answers")
    if m == "sam": return ("Speedup came from shorter,", "corrupted answers")
    if m == "recycling": return ("No gain, and corrupts", "most answers")
    if m == "eagle3": return ("Slower than plain decoding;", "drafter config loads incorrectly")
    if d["ours"] < 1.03: return ("No meaningful gain", "")
    if abs(gap) < 6: return ("Holds up", f"({d['ours']:.2f}× vs {d['paper']:.2f}× claimed)")
    return (f"Real gain, {gap:.0f}% below claim", "")
order = sorted(NAME, key=lambda m: -(data[m]["fair"] or data[m]["ours"]) if data[m]["y"] >= 95 else 10 - data[m]["ours"])
order = [m for m in order if data[m]["y"] >= 95] + [m for m in order if data[m]["y"] < 95]
N = len(order)
fig = plt.figure(figsize=(11.2, 6.9))
gs = fig.add_gridspec(1, 3, width_ratios=[3.3, 1.45, 2.05], wspace=0.06, left=0.135, right=0.985, top=0.80, bottom=0.12)
axS, axQ, axV = (fig.add_subplot(gs[0, i]) for i in range(3))
ys = {m: N - 1 - i for i, m in enumerate(order)}
for ax in (axS, axQ, axV):
    ax.set_ylim(-0.7, N - 0.3); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    for m in order[::2]: ax.axhspan(ys[m] - 0.5, ys[m] + 0.5, color="#f6f6f4", zorder=0, lw=0)
nbroken = sum(data[m]["y"] < 95 for m in order)
for ax in (axS, axQ, axV): ax.axhline(nbroken - 0.5, color=INK3, lw=0.8)
# --- speed panel
axS.set_xlim(0.38, 2.8)
for v in (1.0, 1.5, 2.0, 2.5): axS.axvline(v, color=GRID if v != 1.0 else INK3, lw=0.8 if v != 1.0 else 1.0, ls="-" if v != 1.0 else (0, (2, 3)), zorder=1)
for m in order:
    d, y = data[m], ys[m]; broken = d["y"] < 95; col = RED if broken else BLUE
    end = d["fair"] or d["ours"]
    axS.plot([d["paper"], end], [y, y], color=col, lw=2.2, alpha=0.25, solid_capstyle="round", zorder=2)
    axS.scatter(d["paper"], y, s=62, facecolor=SURF, edgecolor=PAPER, lw=1.8, zorder=3)
    if d["fair"]:
        axS.scatter(d["ours"], y, s=52, facecolor=SURF, edgecolor=col, lw=1.6, zorder=4)
        axS.annotate("own timer", (d["ours"], y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=7, color=INK2)
    axS.scatter(end, y, s=78, color=col, edgecolor=SURF, lw=1.6, zorder=5)
    lo_x = min(d["paper"], end, d["ours"])
    right = end >= d["paper"]
    axS.annotate(f"{end + 1e-6:.2f}×", (end, y), xytext=(9 if right else -9, 0), textcoords="offset points",
                 ha="left" if right else "right", va="center", fontsize=8.4, color=INK, fontweight="bold")
    axS.annotate(f"{d['paper']:.2f}×", (d["paper"], y), xytext=(-9 if right else 9, 0), textcoords="offset points",
                 ha="right" if right else "left", va="center", fontsize=7.8, color=INK3)
    fig.text(0.128, axS.transData.transform((0, y))[1] / fig.dpi / fig.get_size_inches()[1] + 0.008, NAME[m],
             ha="right", va="center", fontsize=9.6, color=INK, fontweight="bold")
    fig.text(0.128, axS.transData.transform((0, y))[1] / fig.dpi / fig.get_size_inches()[1] - 0.016, TYPE[m],
             ha="right", va="center", fontsize=7.3, color=INK3)
axS.set_xticks([1.0, 1.5, 2.0, 2.5], ["1.0×", "1.5×", "2.0×", "2.5×"]); axS.tick_params(length=0, colors=INK2)
axS.set_xlabel("Speedup over greedy decoding (tokens/s)", color=INK2, fontsize=8.5)
axS.set_title("Speed: claimed  →  measured", loc="left", fontsize=10, color=INK, fontweight="bold", pad=8)
# --- integrity panel
axQ.set_xlim(0, 100)
for m in order:
    d, y = data[m], ys[m]; bad = 100 - d["y"]
    axQ.barh(y, 100, height=0.36, color="#e9e8e4", zorder=2, lw=0)
    axQ.barh(y, d["y"], height=0.36, color=BLUE if bad < 5 else "#9fc2ec", zorder=3, lw=0)
    if bad >= 0.05: axQ.barh(y, bad, left=d["y"], height=0.36, color=RED, zorder=3, lw=0)
    axQ.text(100, y + 0.33, f"{bad:.1f}% corrupted" if bad >= 0.05 else "none corrupted", ha="right", va="center",
             fontsize=7.6, color=RED if bad >= 5 else INK2, fontweight="bold" if bad >= 5 else "normal")
axQ.set_xticks([0, 50, 100], ["0", "50", "100%"]); axQ.tick_params(length=0, colors=INK2)
axQ.set_xlabel(f"Share of {n} answers left intact", color=INK2, fontsize=8.5)
axQ.set_title("Answers unchanged?", loc="left", fontsize=10, color=INK, fontweight="bold", pad=8)
# --- verdict panel
axV.set_xlim(0, 1); axV.set_xticks([])
for m in order:
    a, b = verdict(m, data[m]); y = ys[m]; broken = data[m]["y"] < 95
    axV.text(0.04, y + (0.12 if b else 0), a, va="center", fontsize=8.6, color=RED if broken else INK, fontweight="bold" if broken or a == "Holds up" else "normal")
    if b: axV.text(0.04, y - 0.2, b, va="center", fontsize=7.8, color=INK2)
axV.set_title("Verdict", loc="left", fontsize=10, color=INK, fontweight="bold", pad=8)


fig.text(0.016, 0.955, "Claimed vs. measured: ten speculative decoding methods on Qwen2.5-VL-7B", fontsize=15,
         fontweight="bold", color=INK, family="DejaVu Serif")
fig.text(0.016, 0.912, "MMSpec's released code, re-run under one protocol: pure greedy baseline, tokens per second, and a check that the "
         "answer is unchanged.\nThe best intact methods land near 1.57×. The paper's fastest method only leads on a timer that skips image encoding, "
         "and two\nmethods buy their speed with corrupted output.", fontsize=9, color=INK2, va="top", linespacing=1.45)
handles = [plt.Line2D([], [], marker="o", ls="", ms=8, markerfacecolor=SURF, markeredgecolor=PAPER, mew=1.8, label="MMSpec paper (A100)"),
           plt.Line2D([], [], marker="o", ls="", ms=8, color=BLUE, label="Measured here (L4), output intact"),
           plt.Line2D([], [], marker="o", ls="", ms=8, color=RED, label="Measured here, output corrupted"),
           plt.Line2D([], [], marker="o", ls="", ms=7, markerfacecolor=SURF, markeredgecolor=RED, mew=1.6, label="MSD on its own timer")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.005), ncol=4, frameon=False, fontsize=8.3, handletextpad=0.3, columnspacing=1.5)
fig.text(0.985, 0.022, f"{n} questions · L4 · greedy · commit 2904868", ha="right", fontsize=7.3, color=INK3)
fig.savefig(f"{OUT}/fig0_claimed_vs_measured.png", facecolor=SURF)
fig.savefig(f"{OUT}/fig0_claimed_vs_measured.pdf", facecolor=SURF)
print("ok")
