#!/usr/bin/env python3
"""Hourly Jev self-improvement with a consensus gate, an audit trail and automatic rollback.

Jev's weights cannot be trained; what improves is HOW we ask (instructions, criteria, few-shot examples,
ensembles). Every run:
  1. Scoreboard  - measure every production question on dev + test; append to bench/scoreboard.json.
  2. Audit       - a version promoted in the last 72h is re-checked against the one it replaced on the
                   *current* bench; if the old one is now better on test, roll back automatically.
  3. Explore     - for the weakest capabilities, try untried combinations from bench/components.json
                   (+ few-shot examples from the train split, + ensembles of the two best versions).
  4. Consensus   - a candidate is promoted only if ALL hold:
                     a. dev objective beats production by >= MIN_GAIN,
                     b. test objective does not drop by more than TEST_TOLERANCE,
                     c. dev+test accuracy combined is strictly higher,
                     d. a repeat run on dev reproduces the win (not luck).
  5. Log         - every trial, promotion, rejection and rollback goes to bench/audit.jsonl (append-only).
Cost guard: at most RUN_BUDGET Jev calls per run and DAILY_BUDGET per UTC day (bench/budget.json).
"""
import datetime, hashlib, itertools, json, os, random, sys

import jev
import jev_bench as jb

HERE = os.path.dirname(os.path.abspath(__file__))
B = jb.B
MIN_GAIN, TEST_TOLERANCE = 0.02, 0.01
RUN_BUDGET, DAILY_BUDGET = int(os.environ.get("JEV_RUN_BUDGET", 500)), int(os.environ.get("JEV_DAILY_BUDGET", 6000))
NOW = datetime.datetime.now(datetime.timezone.utc)
TS = NOW.isoformat(timespec="seconds")


def load(path, default):
    try:
        return json.load(open(path))
    except (FileNotFoundError, ValueError):
        return default


def save(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def audit(**e):
    with open(B("audit.jsonl"), "a") as f:
        f.write(json.dumps({"ts": TS, "actor": "optimizer", **e}, ensure_ascii=False) + "\n")


def sig(spec):
    return hashlib.md5(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def init_registry():
    reg = {"caps": {}, "rejudge_after": None}
    for cap in jb.CAPS:
        v1 = {**jev.V1[cap], "origin": "v1 (elle yazıldı)", "created": TS}
        reg["caps"][cap] = {"type": jev.TYPES[cap], "prod": ["v1"], "versions": {"v1": v1}, "tried": [sig(jev.V1[cap])],
                            "prev": None, "promoted_at": None}
    return reg


def specs_of(reg, cap, ids):
    return [reg["caps"][cap]["versions"][v] for v in ids]


# ---- candidate generation ---------------------------------------------------------
def with_examples(cap, spec, D, k):
    """Embed k labelled train-split examples per answer into the criteria (few-shot)."""
    if k == 0 or jev.TYPES[cap] == "score":
        return spec
    rows = [r for r in jb.rows_for(D, cap, ("train",))]
    crit = spec["criteria"]

    def ex_text(r, i=None):
        if cap in ("intent", "brand"):
            return r["term"]
        if cap == "risk":
            return r["app"]
        if cap == "kind":
            return f'{r["app"]["name"]}: {r["app"]["desc"]}'
        if cap == "note":
            return r["note"]
        return f'search "{r["term"]}" -> {r["apps"][i]["name"]}: {r["apps"][i]["desc"]}'

    buckets = {}
    if cap == "rival":
        for r in rows:
            for i, y in enumerate(r["labels"]):
                buckets.setdefault("true" if y else "false", []).append(ex_text(r, i))
    else:
        for r in rows:
            key = r["label"] if jev.TYPES[cap] == "choice" else ("true" if r["label"] else "false")
            buckets.setdefault(key, []).append(ex_text(r))
    new = {}
    for opt, desc in crit.items():
        exs = buckets.get(opt, [])[:k]
        new[opt] = {"description": desc, "examples": exs} if exs else desc
    return {**spec, "criteria": new}


def candidates(reg, cap, D, rnd, limit):
    comp = load(B("components.json"), {}).get(cap, {})
    ins, crs = comp.get("instructions", [jev.V1[cap]["instructions"]]), comp.get("criteria", [jev.V1[cap]["criteria"]])
    tried = set(reg["caps"][cap]["tried"])
    pool = []
    for (i, c, k) in itertools.product(range(len(ins)), range(len(crs)), (0, 2, 4)):
        if jev.TYPES[cap] == "score" and k:
            continue
        spec = with_examples(cap, {"instructions": ins[i], "criteria": crs[c]}, D, k)
        if sig(spec) not in tried:
            pool.append((f"talimat#{i} tanım#{c} örnek×{k}", [spec]))
    rnd.shuffle(pool)
    out = pool[:limit]
    # ensemble of production + best other version (consensus of two question wordings)
    c = reg["caps"][cap]
    others = sorted((v for v in c["versions"] if v not in c["prod"]), key=lambda v: -(c["versions"][v].get("dev_obj") or -9))
    if others and len(c["prod"]) == 1:
        ens = c["prod"] + others[:1]
        key = "ensemble:" + "+".join(ens)
        if key not in tried:
            out.append((f"topluluk {'+'.join(ens)}", specs_of(reg, cap, ens)))
    return out


# ---- main loop ---------------------------------------------------------------------
def objective_both(m_dev, m_test):
    return (m_dev["acc"] or 0) * m_dev["n"] + (m_test["acc"] or 0) * m_test["n"]


def main():
    if not jev.api_key():
        sys.exit("TYPESAFE_API_KEY yok")
    D = jb.data()
    reg = load(jev.REGISTRY, None) or init_registry()
    for cap in jb.CAPS:   # new capabilities added later get a v1 automatically
        if cap not in reg["caps"]:
            reg["caps"][cap] = init_registry()["caps"][cap]
    budget = load(B("budget.json"), {})
    day = NOW.strftime("%Y-%m-%d")
    used_today = budget.get(day, 0)
    if used_today >= DAILY_BUDGET:
        audit(action="skip", reason=f"günlük bütçe doldu ({used_today} istek)")
        return
    board = load(B("scoreboard.json"), {"current": {}, "history": []})

    # 1. scoreboard (re-measured only when the questions or the bench changed: same inputs -> same scores)
    fp = sig({"prod": {cap: reg["caps"][cap]["prod"] for cap in jb.CAPS}, "data": sig(D), "model": jev.MODEL})
    cur = {}
    if board.get("fingerprint") == fp and board.get("raw"):
        cur = board["raw"]
    for cap in [] if cur else jb.CAPS:
        c = reg["caps"][cap]
        specs = specs_of(reg, cap, c["prod"])
        d, t = jb.evaluate(cap, specs, ("dev",), D), jb.evaluate(cap, specs, ("test",), D)
        cur[cap] = {"version": "+".join(c["prod"]), "dev": d, "test": t}
        board["history"].append({"ts": TS, "cap": cap, "version": cur[cap]["version"], "dev_acc": d["acc"], "test_acc": t["acc"],
                                 "dev_obj": d["obj"], "test_obj": t["obj"], "n": d["n"] + t["n"]})

    # 2. audit recent promotions; roll back if the replaced version is now better on test
    for cap in jb.CAPS:
        c = reg["caps"][cap]
        if not (c.get("prev") and c.get("promoted_at")):
            continue
        age_h = (NOW - datetime.datetime.fromisoformat(c["promoted_at"])).total_seconds() / 3600
        if age_h > 72:
            c["prev"] = None
            continue
        old_t = jb.evaluate(cap, specs_of(reg, cap, c["prev"]), ("test",), D)
        if old_t["obj"] is not None and cur[cap]["test"]["obj"] is not None and old_t["obj"] > cur[cap]["test"]["obj"] + TEST_TOLERANCE:
            audit(action="rollback", cap=cap, from_version="+".join(c["prod"]), to_version="+".join(c["prev"]),
                  new_test=cur[cap]["test"]["obj"], old_test=old_t["obj"],
                  reason="denetim: önceki sürüm güncel test setinde daha iyi")
            c["prod"], c["prev"], c["promoted_at"] = c["prev"], None, None
            reg["rejudge_after"] = TS

    # 3-4. explore the weakest capabilities, promote by consensus
    rnd = random.Random(NOW.strftime("%Y%m%d%H"))
    weakest = sorted(jb.CAPS, key=lambda cap: (cur[cap]["dev"]["obj"] or 9) + (cur[cap]["test"]["obj"] or 9))
    for cap in weakest:
        if jb.CALLS["n"] >= RUN_BUDGET:
            break
        c = reg["caps"][cap]
        base_d, base_t = cur[cap]["dev"], cur[cap]["test"]
        if base_d["acc"] == 1 and base_t["acc"] == 1 and (base_d.get("brier") or 0) < 0.01:
            continue   # nothing left to win on this bench; the brain should add harder cases
        trials = []
        for label, specs in candidates(reg, cap, D, rnd, limit=4):
            if jb.CALLS["n"] >= RUN_BUDGET:
                break
            m = jb.evaluate(cap, specs, ("dev",), D)
            key = "ensemble:" + label.split(" ")[1] if label.startswith("topluluk") else sig(specs[0])
            c["tried"].append(key)
            trials.append((m["obj"], label, specs, m, key))
            audit(action="trial", cap=cap, candidate=label, dev_obj=m["obj"], dev_acc=m["acc"], prod_dev_obj=base_d["obj"])
        if not trials:
            continue
        best_obj, label, specs, m_dev, key = max(trials, key=lambda x: x[0])
        if best_obj < base_d["obj"] + MIN_GAIN:
            audit(action="reject", cap=cap, candidate=label, reason=f"dev kazancı yetersiz ({best_obj} vs {base_d['obj']})")
            continue
        m_test = jb.evaluate(cap, specs, ("test",), D)
        if m_test["obj"] < base_t["obj"] - TEST_TOLERANCE:
            audit(action="reject", cap=cap, candidate=label, reason=f"test düştü ({m_test['obj']} vs {base_t['obj']})")
            continue
        if objective_both(m_dev, m_test) <= objective_both(base_d, base_t):
            audit(action="reject", cap=cap, candidate=label, reason="dev+test toplam doğruluk artmadı")
            continue
        rep = jb.evaluate(cap, specs, ("dev",), D)
        if rep["obj"] < base_d["obj"] + MIN_GAIN / 2:
            audit(action="reject", cap=cap, candidate=label, reason=f"tekrar çalıştırmada kazanç kayboldu ({rep['obj']})")
            continue
        # promote
        if label.startswith("topluluk"):
            new_prod = key.split(":", 1)[1].split("+")
        else:
            vid = f"v{len(c['versions']) + 1}"
            c["versions"][vid] = {**specs[0], "origin": label, "created": TS, "dev_obj": m_dev["obj"], "test_obj": m_test["obj"]}
            new_prod = [vid]
        audit(action="promote", cap=cap, candidate=label, from_version="+".join(c["prod"]), to_version="+".join(new_prod),
              dev=[base_d["obj"], m_dev["obj"]], test=[base_t["obj"], m_test["obj"]],
              reason="konsensüs: dev kazandı, test düşmedi, toplam doğruluk arttı, tekrar doğruladı")
        c["prev"], c["prod"], c["promoted_at"] = c["prod"], new_prod, TS
        reg["rejudge_after"] = TS
        cur[cap] = {"version": "+".join(new_prod), "dev": m_dev, "test": m_test}

    board["current"] = {cap: {"version": v["version"], "dev_acc": v["dev"]["acc"], "test_acc": v["test"]["acc"],
                              "dev_obj": v["dev"]["obj"], "test_obj": v["test"]["obj"],
                              "brier": v["test"].get("brier"), "n": (v["dev"]["n"] or 0) + (v["test"]["n"] or 0),
                              "versions": len(reg["caps"][cap]["versions"]), "tried": len(reg["caps"][cap]["tried"]),
                              "test_errors": v["test"]["errors"][:5]} for cap, v in cur.items()}
    board["raw"] = cur
    board["fingerprint"] = sig({"prod": {cap: reg["caps"][cap]["prod"] for cap in jb.CAPS}, "data": sig(D), "model": jev.MODEL})
    board["updated"] = TS
    board["history"] = board["history"][-5000:]
    budget[day] = used_today + jb.CALLS["n"]
    save(jev.REGISTRY, reg)
    save(B("scoreboard.json"), board)
    save(B("budget.json"), {k: v for k, v in budget.items() if k >= (NOW - datetime.timedelta(days=30)).strftime("%Y-%m-%d")})
    audit(action="run", calls=jb.CALLS["n"], tokens=jb.CALLS["tokens"], cost_usd=round(jb.CALLS["tokens"] * 0.042 / 1e6, 5))
    print(json.dumps({cap: [v["version"], v["dev_obj"], v["test_obj"]] for cap, v in board["current"].items()}, ensure_ascii=False))
    print("calls", jb.CALLS)


if __name__ == "__main__":
    main()
