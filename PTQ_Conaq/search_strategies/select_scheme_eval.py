"""SelectScheme evaluation on PTQ-NAS-Bench (reproduces Table 8 of the paper).

Usage:  python select_scheme_eval.py [PTQ-NAS-Bench.xlsx]
        Reads the 'Main PTQ-NAS-Bench' sheet of the workbook, which is the
        reference file for all reported results. Requires numpy, pandas and
        openpyxl.

Population: architectures with post-fine-tuning (2-epoch) accuracy and BitOps
for all four schemes. At each budget C, an architecture is included if at least
one scheme satisfies BitOps <= C.

SelectScheme (as evaluated in the paper):
  s_{e,q} = predicted rank of e under Head_q, reversed (larger = better) and
            min-max normalised per scheme over the population;
  among the feasible schemes of e, pick argmax s_{e,q} over the admitted heads
  (Q1-Q3); Q4 is used only when none of Q1-Q3 is feasible.

Regret  = mean (oracle accuracy - policy accuracy), over the architectures
          where the policy is feasible.
Regret0 = same, over the whole population, infeasible choices scored as 0.

Ablation: the same SelectScheme rule is fed different per-scheme
rankings, all as non-dominated front indices over the population:
  QNP proxy targets  (FP32_acc max, ln(1+QNP) min, BitOps min) for Q1-Q3,
                     (ln(QNP) x FP32_acc min, BitOps min) for Q4
                     -- the targets the predictor heads are trained on;
  FP32 + BitOps      (FP32_acc max, BitOps min) -- no QNP;
  ground truth       (post-FT accuracy max, BitOps min).
The raw head outputs (predicted_score) are also reported.
"""
import sys
import numpy as np
import pandas as pd

SCHEMES = {'Q1': 'INT8w8a', 'Q2': 'INT4w4a', 'Q3': 'INT2w8a', 'Q4': 'INT2w2a'}
ADMITTED = [True, True, True, False]      # Q4 is a fallback head
BUDGETS = (0.25, 1.0, 4.0)                # GBitOps
SEED = 0                                  # one generator, used for the budgets in turn
SHEET = 'Main PTQ-NAS-Bench'              # benchmark sheet of the released workbook


def read_benchmark(path):
    """The benchmark table, from the released workbook (or a .csv export of it)."""
    if path.lower().endswith('.csv'):
        return pd.read_csv(path, encoding='utf-8-sig')
    return pd.read_excel(path, sheet_name=SHEET)


def load(path):
    """Columns of the complete-data population, as (n, 4) arrays (FP32 as (n,))."""
    df = read_benchmark(path)
    cols = lambda k: np.column_stack(
        [pd.to_numeric(df[f'{s}_{k}'], errors='coerce').values for s in SCHEMES.values()])
    d = {k: cols(k) for k in ('acc_2epoch', 'GBitOps', 'predicted_rank', 'predicted_score', 'QNP')}
    d['FP32_acc'] = pd.to_numeric(df['FP32_acc'], errors='coerce').values
    complete = ~np.isnan(d['acc_2epoch']).any(1) & ~np.isnan(d['GBitOps']).any(1)
    return {k: v[complete] for k, v in d.items()}


def minmax(score):
    """Per-scheme min-max normalisation (larger = better)."""
    mn, mx = score.min(0), score.max(0)
    return (score - mn) / (mx - mn)


def normalised_scores(rank):
    """Reversed predicted rank, min-max normalised per scheme (1 = best)."""
    return minmax(-rank)


def fronts(objs, step=1000):
    """Non-dominated front index (1 = best) of each row; all objectives minimised."""
    X = np.asarray(objs, float)
    n = len(X)
    D = np.zeros((n, n), bool)                  # D[i, j]: i dominates j
    for s in range(0, n, step):
        a = X[s:s + step, None, :]
        D[s:s + step] = (a <= X[None]).all(-1) & (a < X[None]).any(-1)
    count = D.sum(0)
    rank = np.zeros(n, int)
    cur, r = np.flatnonzero(count == 0), 1
    while len(cur):
        rank[cur] = r
        count = count - D[cur].sum(0)
        count[rank > 0] = -1
        cur, r = np.flatnonzero(count == 0), r + 1
    return rank


def target_fronts(d, kind):
    """Per-scheme front indices of the ablation rankings, shape (n, 4)."""
    fp, qnp, cost, acc = d['FP32_acc'], d['QNP'], d['GBitOps'], d['acc_2epoch']
    out = np.zeros(cost.shape)
    for q in range(4):
        if kind == 'qnp':
            X = (np.c_[np.log1p(qnp[:, q]), -fp, cost[:, q]] if q < 3
                 else np.c_[np.log(qnp[:, q]) * fp, cost[:, q]])
        elif kind == 'fp32':
            X = np.c_[-fp, cost[:, q]]
        else:  # ground truth
            X = np.c_[-acc[:, q], cost[:, q]]
        out[:, q] = fronts(X)
    return out


def selectscheme_regret(acc, cost, score, C):
    """Mean regret of SelectScheme driven by `score` (larger = better) at budget C."""
    feas = cost <= C
    keep = feas.any(1)
    a, f = acc[keep], feas[keep]
    pick = select_scheme(score[keep], f)
    return (np.where(f, a, -np.inf).max(1) - a[np.arange(len(a)), pick]).mean()


def select_scheme(score, feasible):
    """Index of the chosen scheme per architecture (rows must have >=1 feasible)."""
    admitted = feasible & np.array(ADMITTED)
    pool = np.where(admitted.any(1)[:, None], admitted, feasible)
    return np.where(pool, score, -np.inf).argmax(1)


def evaluate(acc, cost, score, C, rng):
    feas = cost <= C
    keep = feas.any(1)
    a, f, c, s = acc[keep], feas[keep], cost[keep], score[keep]
    n, ix = len(a), np.arange(len(a))
    oracle_acc = np.where(f, a, -np.inf).max(1)
    rows = []

    def picked(name, pick):
        pa = a[ix, pick]
        r = (oracle_acc - pa).mean()
        rows.append((name, 100.0, pa.mean(), c[ix, pick].mean(), r, r,
                     [np.mean(pick == q) * 100 for q in range(4)]))

    picked('Oracle', np.where(f, a, -np.inf).argmax(1))
    for q, name in enumerate(SCHEMES):
        v = f[:, q]
        reg0 = (oracle_acc - np.where(v, a[:, q], 0)).mean()
        mix = [100.0 if j == q else 0.0 for j in range(4)]
        if v.any():
            rows.append((f'always-{name}', v.mean() * 100, a[v, q].mean(), c[v, q].mean(),
                         (oracle_acc[v] - a[v, q]).mean(), reg0, mix))
        else:
            rows.append((f'always-{name}', 0.0, np.nan, np.nan, np.nan, reg0, mix))
    picked('random', np.array([rng.choice(np.flatnonzero(r)) for r in f]))
    q2_else_q4 = np.where(f[:, 1], 1, 3)
    picked('Q2, else Q4', q2_else_q4)
    picked('most exp.', np.where(f[:, 0], 0, q2_else_q4))   # Q1 > Q2 > Q4
    picked('SelectScheme', select_scheme(s, f))
    return n, rows


def main(path):
    d = load(path)
    acc, cost = d['acc_2epoch'], d['GBitOps']
    score = normalised_scores(d['predicted_rank'])
    rng = np.random.default_rng(SEED)
    print(f'complete-data architectures: {len(acc):,}')
    for C in BUDGETS:
        n, rows = evaluate(acc, cost, score, C, rng)
        print(f'\nC = {C} GBitOps - {n:,} feasible architectures')
        print(f'{"Policy":14s} {"Valid%":>7s} {"Acc":>7s} {"GBitOps":>8s} {"Regret":>7s} {"Regret0":>8s}  Q1/Q2/Q3/Q4')
        for name, v, ac, gb, r, r0, mix in rows:
            fmt = lambda x, d: '---' if np.isnan(x) else f'{x:.{d}f}'
            print(f'{name:14s} {v:7.1f} {fmt(ac, 2):>7s} {fmt(gb, 3):>8s} {fmt(r, 2):>7s} {r0:8.2f}  '
                  + '/'.join(f'{m:.0f}' for m in mix))

    print('\nAblation: ranking fed to the same SelectScheme rule, mean regret')
    rankings = {
        'released heads (ranks)': score,
        'released heads (raw scores)': minmax(d['predicted_score']),
        'QNP proxy targets': normalised_scores(target_fronts(d, 'qnp')),
        'FP32 + BitOps (no QNP)': normalised_scores(target_fronts(d, 'fp32')),
        'ground-truth fronts': normalised_scores(target_fronts(d, 'gt')),
    }
    print(f'{"Ranking":28s}' + ''.join(f'{C:>8}' for C in BUDGETS))
    for name, s in rankings.items():
        print(f'{name:28s}' + ''.join(f'{selectscheme_regret(acc, cost, s, C):8.2f}' for C in BUDGETS))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'PTQ-NAS-Bench.xlsx')
