#!/usr/bin/env python3
"""Jev scoreboard: measure any question version of any capability on the hand-labelled bench (bench/data.json).

Metrics per capability: accuracy, Brier score (calibration; lower is better) and
objective = accuracy - 0.5 * Brier  (score dims: pairwise ordering accuracy + 0.05 * mean margin).
Run `python3 jev_bench.py` to print the current production scoreboard.
"""
import json, os, statistics
from concurrent.futures import ThreadPoolExecutor

import jev

HERE = os.path.dirname(os.path.abspath(__file__))
B = lambda f: os.path.join(HERE, "bench", f)
CALLS = {"n": 0, "tokens": 0}
CAPS = ["intent", "brand", "rival", "risk", "audience", "pay", "build", "kind", "note"]


def data():
    return json.load(open(B("data.json")))


def ask(state, q):
    r = jev.ask(state, q)
    CALLS["n"] += 1
    CALLS["tokens"] += r["usage"]["input_tokens"]
    return r["answers"]


def pmap(fn, xs, n=8):
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(fn, xs))


def rows_for(D, cap, splits):
    if cap in ("audience", "pay", "build"):
        return [r for r in D["pairs"] if r["dim"] == cap and r["split"] in splits]
    return [r for r in D[cap] if r["split"] in splits]


def _choice(ans, label):
    p = ans["probabilities"]
    return ans["choice"] == label, sum((v - (k == label)) ** 2 for k, v in p.items())


def _noul(ans, label):
    p = ans["noul"]
    return (p >= 0.5) == bool(label), (p - float(bool(label))) ** 2


def evaluate(cap, specs, splits=("dev",), D=None):
    """specs: list of version specs (several = ensemble, answers averaged). Returns metrics + errors."""
    D = D or data()
    rows = rows_for(D, cap, splits)
    if not rows:
        return {"n": 0, "acc": None, "brier": None, "obj": None, "errors": []}
    res = []   # (correct, brier, detail)

    if cap in ("intent", "brand"):
        def one(r):
            q = {}; jev.add_q(q, "x", cap, specs)
            a = jev.merged(ask(jev.keyword_state(r["term"], None, r.get("top", [])), q), "x")
            ok, br = _choice(a, r["label"]) if cap == "intent" else _noul(a, r["label"])
            return ok, br, (r["term"], r["label"], a.get("choice", round(a.get("noul", 0), 2)))
        res = pmap(one, rows)
    elif cap == "rival":
        def one(r):
            q = {}
            for i in range(len(r["apps"])):
                jev.add_q(q, f"r{i}", "rival", specs, i=i)
            ans = ask(jev.keyword_state(r["term"], None, r["apps"]), q)
            out = []
            for i, y in enumerate(r["labels"]):
                a = jev.merged(ans, f"r{i}")
                ok, br = _noul(a, y)
                out.append((ok, br, (r["term"], r["apps"][i]["name"], y, round(a["noul"], 2))))
            return out
        res = [x for g in pmap(one, rows) for x in g]
    elif cap == "risk":
        def one(r):
            q = {}; jev.add_q(q, "x", "risk", specs, subject=jev.SUBJECT_APP)
            a = jev.merged(ask(jev.app_state({"name": "App", "desc": r["app"]}), q), "x")
            ok, br = _noul(a, r["label"])
            return ok, br, (r["app"][:60], r["label"], round(a["noul"], 2))
        res = pmap(one, rows)
    elif cap == "kind":
        def one(r):
            q = {}; jev.add_q(q, "x", "kind", specs)
            a = jev.merged(ask(jev.app_state(r["app"]), q), "x")
            ok, br = _choice(a, r["label"])
            return ok, br, (r["app"]["name"], r["label"], a["choice"])
        res = pmap(one, rows)
    elif cap == "note":
        def one(r):
            q = {}; jev.add_q(q, "x", "note", specs)
            a = jev.merged(ask({"note": r["note"]}, q), "x")
            ok, br = _choice(a, r["label"])
            return ok, br, (r["note"], r["label"], a["choice"])
        res = pmap(one, rows)
    else:   # score dims: ordering of pairs
        terms = sorted({t for r in rows for t in (r["hi"], r["lo"])})

        def one(t):
            q = {}; jev.add_q(q, "x", cap, specs, subject=jev.SUBJECT_KW)
            return t, jev.norm(jev.merged(ask(jev.keyword_state(t, None, []), q), "x"))
        val = dict(pmap(one, terms))
        margins = [val[r["hi"]] - val[r["lo"]] for r in rows]
        acc = sum(m > 0 for m in margins) / len(rows)
        errors = [(r["hi"], r["lo"], round(m, 2)) for r, m in zip(rows, margins) if m <= 0]
        return {"n": len(rows), "acc": round(acc, 4), "brier": None, "margin": round(statistics.mean(margins), 4),
                "obj": round(acc + 0.05 * statistics.mean(margins), 4), "errors": errors}

    acc = sum(r[0] for r in res) / len(res)
    brier = statistics.mean(r[1] for r in res)
    return {"n": len(res), "acc": round(acc, 4), "brier": round(brier, 4), "obj": round(acc - 0.5 * brier, 4),
            "errors": [r[2] for r in res if not r[0]]}


if __name__ == "__main__":
    D = data()
    for cap in CAPS:
        specs = jev.prod_specs(cap)
        d, t = evaluate(cap, specs, ("dev",), D), evaluate(cap, specs, ("test",), D)
        print(f"{cap:9} dev acc {d['acc']} obj {d['obj']} (n={d['n']}) | test acc {t['acc']} obj {t['obj']} (n={t['n']})"
              + (f" | test errors: {t['errors'][:3]}" if t["errors"] else ""))
    print("calls", CALLS)
