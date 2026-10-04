#!/usr/bin/env python3
"""Round 2: hard, real-world tests on App Scout's own mined data. Run: python3 jev_eval2.py"""
import json, math, statistics
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import jev
from jev_eval import ask, pmap, USAGE, run_rivals, app

R = {}
KW = json.load(open("keywords.json"))

# ------------------------------------------------ 1. brand vs descriptive, on real mined terms (hand labels)
# brand = a proper name of one company/product/app; need = plain words for a need (even if an app uses them as title)
LABELS = {
    "chiropractic ethics ca": "need", "welding written - passrun": "brand", "cswip welding inspector": "need",
    "welding symbols": "need", "physical therapy journal": "need", "speech therapy delhi": "brand",
    "rccs sonography exam prep 2026": "need", "rmsks sonography exam prep": "need",
    "dental assistant - dental plan": "brand", "pharmacy technician letter": "brand", "paramedic coach": "brand",
    "student midwife baby catches": "need", "paramedic nrp exam prep": "need", "paramedic registration prep uk": "need",
    "paramedic survey helper": "need", "paramedic airway review": "need", "paramedic case tracker": "need",
    "paramedic special pops review": "need", "guide to clinical midwifery": "brand", "amcb cm midwifery exam prep": "need",
    "smart optometry": "brand", "audiology clinic training": "need", "audiology a.i.": "brand",
    "massage therapy notes - needle": "brand", "cmtbc massage therapy prep": "need",
    "align functional movement & massage therapy": "brand", "acupuncture tools": "need",
    "aws cws welding exam prep": "need", "woodcheck: carpentry checklist": "brand", "machinist journeyman": "need",
}
EXTRA_BRANDS = ["foreflight", "hinge health", "miller weld setting calculator", "picsart", "medscape", "epocrates"]

BRAND_Q = {"type": "noul",
           "instructions": "Is `search_term` a proper name - the brand, product, company or app name of one specific "
                           "app or company - rather than plain words describing what the searcher needs? "
                           "Use `top_results` as evidence: a brand usually appears as the distinctive title of one result.",
           "criteria": {"true": "A proper or brand name (like 'ForeFlight', 'PassRun', 'Smart Optometry'), possibly with generic words added",
                        "false": "Plain descriptive words for a need or topic, even if one app happens to use these words as its title "
                                 "(like 'paramedic exam prep', 'welding symbols', 'acupuncture tools')"}}


def top_state(term):
    k = KW.get(term)
    tops = k["top"] if k else []
    return {"search_term": term, "top_results": [{"rank": i + 1, "name": a["name"], "ratings": a["ratings"],
                                                  "description": (a.get("desc") or "")[:200]} for i, a in enumerate(tops)]}


def t_brand():
    cases = list(LABELS.items()) + [(b, "brand") for b in EXTRA_BRANDS]

    def one(c):
        term, y = c
        st = top_state(term)
        a = ask(st, {"intent": {"type": "choice", "criteria": jev.INTENT,
                                "instructions": "What is a person who types `search_term` into the App Store search box looking for?"},
                     "brand": BRAND_Q})
        bare = ask({"search_term": term}, {"brand": BRAND_Q})["brand"]["noul"]
        return term, y, a["intent"]["choice"], a["brand"]["noul"], bare
    rows = pmap(one, cases)
    old = [(r[2] in ("brand", "local_business")) == (r[1] == "brand") for r in rows]
    new = [(r[3] >= 0.5) == (r[1] == "brand") for r in rows]
    bare = [(r[4] >= 0.5) == (r[1] == "brand") for r in rows]
    R["brand"] = {"n": len(rows), "old_intent_acc": sum(old) / len(rows), "new_brand_q_acc": sum(new) / len(rows),
                  "new_no_evidence_acc": sum(bare) / len(rows),
                  "new_errors": [(r[0], r[1], round(r[3], 2)) for r, ok in zip(rows, new) if not ok],
                  "old_errors": [(r[0], r[1], r[2]) for r, ok in zip(rows, old) if not ok]}


# ------------------------------------------------ 2. do Jev's app features predict the user's own good/bad labels?
def auc(scores, ys):
    pos = [s for s, y in zip(scores, ys) if y]; neg = [s for s, y in zip(scores, ys) if not y]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / max(1, len(pos) * len(neg))


def t_user_labels():
    import server
    labels = json.load(open("labels.json")); ja = json.load(open("jev_apps.json"))
    ids = [i for i in labels if i in ja]
    X = np.array([jev.app_vector(ja[i]) for i in ids]); y = np.array([labels[i] == "good" for i in ids], float)
    # leave-one-out: fit on n-1, score the held-out app
    loo = []
    for k in range(len(ids)):
        m = np.ones(len(ids), bool); m[k] = False
        w, b = server.fit_jev_weights(X[m], y[m], np.ones(m.sum()))
        loo.append(float(X[k] @ w + b))
    prior = list(X @ server.JEV_PRIOR + server.JEV_BIAS)
    single = {name: auc(list(X[:, j]), list(y)) for j, name in enumerate(["audience", "pay", "build", "risk"])}
    R["user_labels"] = {"n": len(ids), "good": int(y.sum()), "auc_prior_only": round(auc(prior, list(y)), 3),
                        "auc_learned_loo": round(auc(loo, list(y)), 3), "auc_single_feature": {k: round(v, 3) for k, v in single.items()}}


# ------------------------------------------------ 3. knowledge boundary: can Jev estimate search popularity?
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def t_popularity():
    terms = [k for k in KW.values() if k.get("foundAt")][:80]

    def one(k):
        a = ask({"search_term": k["term"]}, {"pop": {"type": "score", "instructions": "How many people search the US App Store for `search_term` each month?",
                                                     "criteria": ["Almost nobody (under 100)", "A few hundred", "A few thousand", "Tens of thousands", "Hundreds of thousands or more"]}})
        return jev.norm(a["pop"])
    est = pmap(one, terms)
    proxy = [k["pop"] for k in terms]
    R["popularity"] = {"n": len(terms), "spearman_vs_autocomplete_proxy": round(spearman(est, proxy), 3)}


# ------------------------------------------------ 4. Turkish market: Turkish term + Turkish descriptions
TR = {"kaynak sembolleri": [
    (app("Kaynak Sembolleri Rehberi", "AWS ve ISO kaynak sembollerinin resimli anlamları ve sınav için test."), 1, "direct"),
    (app("Kaynak Hesaplayıcı", "MIG ve TIG kaynak için amper ve tel besleme hızı hesaplama."), 0, "adjacent"),
    (app("Fener", "En parlak LED el feneri."), 0, "unrelated"),
    (app("Sembol Klavyesi", "Mesajlarınız için yüzlerce sembol ve emoji."), 0, "mention")],
    "akupunktur noktaları": [
    (app("Akupunktur Atlası", "361 akupunktur noktasının 3D vücut üzerinde yeri, meridyenler ve endikasyonlar."), 1, "direct"),
    (app("Masaj Randevu", "Masaj terapistleri için randevu ve müşteri notları."), 0, "adjacent"),
    (app("Yemek Tarifleri", "Binlerce Türk yemeği tarifi."), 0, "unrelated")]}


def t_turkish():
    rows = []
    for term, items in TR.items():
        ps = run_rivals(term, items)
        rows += [(term, it[0]["name"], it[1], p) for it, p in zip(items, ps)]
    intent = [ask({"search_term": t}, {"i": {"type": "choice", "criteria": jev.INTENT,
              "instructions": "What is a person who types `search_term` into the App Store search box looking for?"}})["i"]["choice"]
              for t in ["kaynak sembolleri", "akupunktur noktaları", "ehliyet sınav soruları", "trendyol", "acıbadem hastanesi"]]
    R["turkish"] = {"rival_acc": sum((p >= 0.5) == bool(y) for *_, y, p in rows) / len(rows),
                    "rival_rows": [(r[1], r[2], round(r[3], 2)) for r in rows],
                    "intent": dict(zip(["kaynak sembolleri (need)", "akupunktur noktaları (need)", "ehliyet sınav soruları (need)",
                                        "trendyol (brand)", "acıbadem hastanesi (local)"], intent))}


# ------------------------------------------------ 5. borderline rivals: partial overlap ("suite" apps that include the need)
def t_partial():
    term = "welding symbols"
    items = [
        (app("Welder's Handbook", "All-in-one welding app: calculators, electrode charts, and a complete welding symbols chapter with quizzes."), "suite (symbols is a full feature)"),
        (app("CWI Exam Prep", "Practice exams for the Certified Welding Inspector test, including questions on welding symbols."), "exam prep that covers symbols"),
        (app("Blueprint Reading for Trades", "Learn to read construction and fabrication drawings, including welding symbols."), "broader topic"),
    ]
    ps = run_rivals(term, [(a, 0, "") for a, _ in items])
    R["partial"] = [(d, round(p, 2)) for (_, d), p in zip(items, ps)]


if __name__ == "__main__":
    for f in (t_brand, t_user_labels, t_popularity, t_turkish, t_partial):
        f(); print(f.__name__, "✓", flush=True)
    R["usage"] = {"calls": USAGE["calls"], "input_tokens": USAGE["in"], "cost_usd": round(USAGE["in"] * 0.042 / 1e6, 4)}
    json.dump(R, open("jev_eval2_results.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
