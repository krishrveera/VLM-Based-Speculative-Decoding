"""Paper figures for one run. Usage:
    python tools/make_figures.py runs/l4_test_600 analysis/l4_test_600/summary_vs_greedy.csv reports/figures
Writes fig1..fig6 as PNG and PDF."""
import csv, json, os, random, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUN, CSV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)

# palette (validated categorical order + status colors), light surface
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
CRIT, SERIOUS = "#d03b3b", "#ec835a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 220, "legend.frameon": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left"})

NAME = {"eagle": "EAGLE-1", "eagle2": "EAGLE-2", "eagle3": "EAGLE-3", "vispec": "ViSpec", "msd_e2e": "MSD",
        "medusa": "Medusa", "pld": "PLD", "lookahead": "Lookahead", "sam": "SAM", "recycling": "Recycling"}
TASK = [("GQA", "General\nVQA"), ("TVQA", "Text\nVQA"), ("CQA", "Chart\nVQA"), ("IC", "Image\ncaptioning"),
        ("CR", "Complex\nreasoning"), ("MTC", "Multi-turn\nchat")]
TOPIC = {"general vqa": "GQA", "text vqa": "TVQA", "chart understanding": "CQA", "image captioning": "IC",
         "complex reasoning pro": "CR", "multi-turn conversation": "MTC"}

rows = list(csv.DictReader(open(CSV)))
S = {(r["method"], r["topic"]): r for r in rows}
f = lambda m, t, k: float(S[(m, t)][k])

def load(m): return {json.loads(l)["question_id"]: json.loads(l) for l in open(f"{RUN}/{m}-temperature-0.jsonl") if l.strip()}
G = load("baseline_greedy"); R = {m: load(m) for m in NAME}; Q = sorted(G); n = len(Q)

def tok(m, q): return sum((G if m == "g" else R[m])[q]["choices"][0]["new_tokens"])
def tim(m, q, e2e=False):
    c = (G if m == "g" else R[m])[q]["choices"][0]
    return sum(c["wall_time"]) + (sum(c.get("embed_prep_time", [0])) if e2e else 0)
def spd(m, idx, e2e=False):
    b = sum(tok("g", Q[i]) for i in idx) / sum(tim("g", Q[i]) for i in idx)
    return sum(tok(m, Q[i]) for i in idx) / sum(tim(m, Q[i], e2e) for i in idx) / b
random.seed(0)
boots = [[random.randrange(n) for _ in range(n)] for _ in range(1000)]
def ci(m, e2e=False):
    v = sorted(spd(m, b, e2e) for b in boots); return v[25], v[974]

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); plt.close(fig)

# corruption classification (same rules as analyze_runs.py)
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
corr = {m: [classify(m, q) for q in Q] for m in NAME}
mid_pct = {m: 100 * sum(c[1] for c in corr[m]) / n for m in NAME}
tail_pct = {m: 100 * sum(c[0] and not c[1] for c in corr[m]) / n for m in NAME}
BROKEN = {m for m in NAME if mid_pct[m] >= 5}

# ---------- Fig 1: paper vs ours ----------
order = sorted(NAME, key=lambda m: f(m, "Overall", "speedup_tps"))
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for i, m in enumerate(order):
    ours = f(m, "Overall", "speedup_tps"); paper = float(S[(m, "Overall")]["paper_speedup"]); lo, hi = ci(m)
    ax.plot([min(ours, paper), max(ours, paper)], [i, i], color=GRID, lw=2.2, zorder=1)
    ax.scatter(paper, i, s=58, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.errorbar(ours, i, xerr=[[ours - lo], [hi - ours]], fmt="o", ms=7.5, color=BLUE, mec=SURF, mew=1.5, capsize=2.5, lw=1.4, zorder=4)
    lab = f"{ours + 1e-9:.2f}×"
    ax.text(max(ours, paper) + 0.06, i, lab, va="center", fontsize=8.5, color=INK)
    if m in BROKEN: ax.text(max(ours, paper) + 0.38, i, f"corrupts {mid_pct[m]:.0f}% of answers", va="center", fontsize=8, color=CRIT)
    if m == "msd_e2e":
        lo2, hi2 = ci(m, True); fair = spd(m, range(n), True)
        ax.errorbar(fair, i - 0.32, xerr=[[fair - lo2], [hi2 - fair]], fmt="D", ms=5.5, color=BLUE, mfc=SURF, mew=1.4, capsize=2, lw=1.1, zorder=4)
        ax.text(fair - 0.05, i - 0.32, f"timed fairly {fair:.2f}×", va="center", ha="right", fontsize=8, color=INK2)
ax.axvline(1.0, color=INK2, lw=1, ls=(0, (3, 3)))
ax.text(1.01, -0.75, "no speedup", fontsize=8, color=INK2, va="center")
ax.set_yticks(range(len(order)), [NAME[m] for m in order]); ax.grid(axis="y", visible=False)
ax.set_xlim(0.5, 3.15); ax.set_ylim(-1.05, len(order) - 0.5); ax.set_xlabel("Speedup over greedy decoding (tokens per second)")
ax.set_title("Published speedups vs. a fair re-run")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=7, label="MMSpec paper (A100)"),
                   plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=7, label="Ours, 95% CI (L4)")],
          loc="lower right", fontsize=8.5)
save(fig, "fig1_paper_vs_ours_speedup")

# ---------- Fig 2: answer problems ----------
order2 = sorted(NAME, key=lambda m: mid_pct[m] + tail_pct[m])
fig, ax = plt.subplots(figsize=(7.2, 4.2))
y = range(len(order2))
ax.barh(y, [mid_pct[m] for m in order2], height=0.62, color=CRIT, edgecolor=SURF, linewidth=2, label="Corrupted mid-answer")
ax.barh(y, [tail_pct[m] for m in order2], left=[mid_pct[m] for m in order2], height=0.62, color=SERIOUS, edgecolor=SURF, linewidth=2,
        label="Stray token after a complete answer")
for i, m in enumerate(order2):
    tot = mid_pct[m] + tail_pct[m]
    ax.text(tot + 0.8, i, f"{mid_pct[m]:.1f}% + {tail_pct[m]:.1f}%" if tail_pct[m] >= 1 else f"{mid_pct[m]:.1f}%", va="center", fontsize=8.5, color=INK)
ax.set_yticks(list(y), [NAME[m] for m in order2]); ax.grid(axis="y", visible=False)
ax.set_xlim(0, 75); ax.set_xlabel(f"Share of the {n} answers (%)  ·  greedy baseline: 0%")
ax.set_title("Speculative decoding should not change the answer")
ax.legend(loc="lower right", fontsize=8.5)
save(fig, "fig2_answer_problems")

# ---------- Fig 3: clock time vs tokens/s ----------
order3 = sorted(NAME, key=lambda m: f(m, "Overall", "speedup_time"))
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for i, m in enumerate(order3):
    a, b = f(m, "Overall", "speedup_tps"), f(m, "Overall", "speedup_time")
    ax.plot([a, b], [i, i], color=GRID, lw=2.2, zorder=1)
    ax.scatter(b, i, s=55, color=GRAY, zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.scatter(a, i, s=62, color=BLUE, zorder=4, edgecolor=SURF, linewidth=1.5)
    r = f(m, "Overall", "tokens_ratio")
    ax.text(max(a, b) + 0.05, i, f"writes {r:.2f}× the tokens" if abs(r - 1) > 0.05 else "", va="center", fontsize=8.2, color=INK2)
ax.axvline(1.0, color=INK2, lw=1, ls=(0, (3, 3)))
ax.set_yticks(range(len(order3)), [NAME[m] for m in order3]); ax.grid(axis="y", visible=False)
ax.set_xlim(0.5, 2.75); ax.set_xlabel("Speedup over greedy decoding")
ax.set_title("Shorter answers look fast on the clock")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=GRAY, ms=7, label="Wall-clock time"),
                   plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=7, label="Tokens per second")], loc="lower right", fontsize=8.5)
save(fig, "fig3_clock_vs_tokens_per_second")

# ---------- Fig 4: MSD timer per task ----------
own, fair, emb = [], [], []
for code, _ in TASK:
    qs = [i for i, q in enumerate(Q) if TOPIC[G[q]["topic"]] == code]
    own.append(spd("msd_e2e", qs)); fair.append(spd("msd_e2e", qs, True))
    e = sorted(x for i in qs for x in R["msd_e2e"][Q[i]]["choices"][0]["embed_prep_time"]); emb.append(e[len(e) // 2])
ref = [f("eagle", c, "speedup_tps") for c, _ in TASK]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
x = range(len(TASK)); w = 0.36
ax.bar([i - w / 2 for i in x], own, w, color=GRAY, edgecolor=SURF, linewidth=2, label="MSD, its own timer")
ax.bar([i + w / 2 for i in x], fair, w, color=BLUE, edgecolor=SURF, linewidth=2, label="MSD, vision encoder included")
ax.scatter(list(x), ref, marker="_", s=420, color=ORANGE, linewidth=2.4, zorder=5, label="EAGLE-1 (for reference)")
for i in x:
    ax.text(i + w / 2, fair[i] - 0.06, f"{fair[i]:.2f}", ha="center", va="top", fontsize=8, color="white", fontweight="bold")
    ax.text(i - w / 2, own[i] - 0.06, f"{own[i]:.2f}", ha="center", va="top", fontsize=8, color="white", fontweight="bold")
    ax.text(i, -0.36, f"{emb[i]:.2f} s", ha="center", va="top", fontsize=7.8, color=INK2)
ax.axhline(1.0, color=INK2, lw=1, ls=(0, (3, 3)))
ax.set_xticks(list(x), [t for _, t in TASK]); ax.tick_params(axis="x", pad=2)
ax.set_ylim(0, 2.2); ax.set_ylabel("Speedup (tokens per second)"); ax.grid(axis="x", visible=False)
ax.text(-0.62, -0.36, "vision encoder\nper turn:", ha="right", va="top", fontsize=7.6, color=INK2)
ax.set_title("MSD's timer skips the vision encoder")
ax.legend(loc="upper left", fontsize=8.2, ncol=3)

save(fig, "fig4_msd_timer_by_task")

# ---------- Fig 5: per-task heatmap ----------
hm = ["vispec", "eagle", "eagle2", "medusa", "lookahead", "pld", "sam", "recycling", "eagle3"]
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
Z = np.array([[f(m, c, "speedup_tps") for c, _ in TASK] for m in hm])
cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
fig, ax = plt.subplots(figsize=(7.4, 4.5))
im = ax.imshow(Z, cmap=cmap, norm=TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0), aspect="auto")
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        v = Z[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if (v > 1.65 or v < 0.68) else INK)
ax.set_xticks(range(len(TASK)), [t for _, t in TASK]); ax.set_yticks(range(len(hm)), [NAME[m] + ("*" if m in BROKEN else "") for m in hm])
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
ax.set_xticks(np.arange(-.5, len(TASK)), minor=True); ax.set_yticks(np.arange(-.5, len(hm)), minor=True)
ax.grid(which="minor", color=SURF, linewidth=2); ax.tick_params(which="minor", length=0); ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False); cb.set_label("Speedup (tokens/s)", color=INK2)
ax.set_title("Speedup by task")
fig.text(0.01, -0.01, "* corrupts more than 5% of answers. MSD omitted here (see Fig. 4).", fontsize=7.8, color=INK2)
save(fig, "fig5_speedup_by_task")

# ---------- Fig 6: MAT ours vs paper ----------
fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.plot([0, 2.9], [0, 2.9], color=INK2, lw=1, ls=(0, (3, 3)))
off = {"eagle": (6, -11), "eagle2": (6, -4), "msd_e2e": (-34, 5), "vispec": (-8, 7), "eagle3": (6, -3), "medusa": (6, -4),
       "lookahead": (6, 2), "sam": (6, -8), "pld": (-24, 4), "recycling": (6, -6)}
LOW = {"lookahead": (0.62, 0.55), "sam": (0.62, 0.40), "pld": (0.62, 0.26), "eagle3": (0.62, 0.12), "recycling": (0.62, -0.02)}
for m in NAME:
    p, o = float(S[(m, "Overall")]["paper_MAT"]), f(m, "Overall", "MAT")
    far = m in ("vispec", "eagle3")
    ax.scatter(p, o, s=52, color=ORANGE if far else BLUE, edgecolor=SURF, linewidth=1.5, zorder=3)
    if m in LOW: ax.annotate(NAME[m], (p, o), xytext=LOW[m], fontsize=8.2, color=INK, va="center",
                             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=4))
    else: ax.annotate(NAME[m], (p, o), textcoords="offset points", xytext=off[m], fontsize=8.2, color=INK)
ax.set_xlim(0, 2.9); ax.set_ylim(-0.1, 2.9); ax.set_aspect("equal")
ax.set_xlabel("MAT reported by MMSpec"); ax.set_ylabel("MAT measured here")
ax.set_title("Acceptance reproduces for 8 of 10")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=7, label="matches the paper"),
                   plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=7, label="does not match")], loc="lower right", fontsize=8.5)
save(fig, "fig6_mat_ours_vs_paper")
print("wrote figures to", OUT)
