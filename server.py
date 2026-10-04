#!/usr/bin/env python3
"""App Scout: local review UI + learning loop + niche finder. Run: python3 server.py

Ranking  = semantic similarity (macOS NLEmbedding word vectors, IDF-weighted) to liked vs. disliked apps
           + Naive Bayes on words, both reason-weighted, blended with a utility-style prior,
           minus penalties for clones of liked apps and for crowded niches.
Discover = Apple's own "You Might Also Like" graph of liked apps + App Store autocomplete + small-dev portfolios.
Niches   = autocomplete prefix test (popularity proxy) x weakness of the current top-10.
"""
import base64, datetime, json, math, os, plistlib, random, re, shutil, statistics, subprocess, sys, threading, time
import urllib.parse, urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import jev

from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda f: os.path.join(HERE, f)                       # code / static files
DATA_DIR = os.environ.get("DATA_DIR", HERE)               # cloud: a persistent volume, e.g. /data
DP = lambda f: os.path.join(DATA_DIR, f)                  # everything the app writes
LOCK = threading.Lock()
TODAY = datetime.date.today()

STOP = set("the and for you your with that this are can will our app from have all not but use more has any one get "
           "its out new now just also into they them their than then what when who how why per was were been "
           "which these those other such only over very each both able about after before between while would "
           "should could may might must make made like free premium subscription terms privacy policy "
           "apple itunes account payment charged period cancel renew automatically least hours current "
           "iphone ipad device phone apps features feature easy simple best want need help every any".split())
NAME_STOP = STOP | set("pro lite app apps plus free best ai tool tools master simple smart easy".split())

# Reasons the user can give for "bad". Weight = how much it says the *idea/niche* is bad.
# "Kalitesiz / eski" is about this listing, not the idea: a weak competitor is an opportunity, so it teaches nothing negative.
REASON_W = {"Alakasız": 1.0, "Para yapmaz": 1.0, "Sıkıcı fikir": 1.0, "Ret riski": 1.0, "Yapması zor": 0.7,
            "Çok rekabet / doygun": 0.3, "Çok benzer": 0.2, "Kalitesiz / eski": 0.0}
IDEA_TERMS = ["magnifier", "metal detector", "hidden camera detector", "decibel meter", "white noise", "qibla finder",
              "anti theft alarm", "breathing exercise", "sound amplifier", "fasting tracker", "countdown widget",
              "jet lag", "ok to wake clock", "silent alarm", "battery alert", "guitar tuner", "birthday reminder",
              "speaker cleaner", "stud finder", "reading light"]
UTIL_GENRES = {"utilities", "productivity", "health & fitness", "photo & video", "weather", "navigation",
               "lifestyle", "business", "finance", "education", "music", "graphics & design", "reference"}
UTIL_KW = ["scanner", "tracker", "meter", "detector", "finder", "converter", "widget", "timer", "compass",
           "measure", "recorder", "booster", "editor", "monitor", "maker", "level", "magnifier", "sounds",
           "counter", "reminder", "identifier", "test", "map", "forecast", "alarm", "clock", "calculator"]

CLONE_SIM = 0.80   # cosine to a liked app above this = same niche, not a new idea
CLUSTER_SIM = 0.78  # apps this close are grouped as one niche (feed shows max 2 per niche)


# ---- storage ---------------------------------------------------------------
def load(name, default):
    try:
        return json.load(open(DP(name)))
    except Exception:
        return default


def save(name, obj):
    tmp = DP(name) + ".tmp"
    json.dump(obj, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, DP(name))


JOB = {"running": False, "kind": "", "progress": "", "result": None}


def progress(msg):
    JOB["progress"] = msg
    if "--task" in sys.argv:
        print(msg, flush=True)


# ---- App Store access (all free/public) ------------------------------------
def http_get(url, headers=None, tries=4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
            return urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            time.sleep(3 * (attempt + 1))
    return b""


def http_json(url):
    try:
        return json.loads(http_get(url) or b"{}")
    except Exception:
        return {}


def search(term, limit=25):
    q = urllib.parse.urlencode({"term": term, "entity": "software", "limit": limit, "country": "us"})
    r = [x for x in http_json("https://itunes.apple.com/search?" + q).get("results", []) if x.get("kind") == "software"]
    time.sleep(1.2)
    return r


def lookup(ids):
    out = []
    ids = list(ids)
    for i in range(0, len(ids), 100):
        r = http_json("https://itunes.apple.com/lookup?country=us&id=" + ",".join(map(str, ids[i:i + 100])))
        out += [x for x in r.get("results", []) if x.get("kind") == "software"]
        time.sleep(1.2)
    return out


def dev_apps(dev_id):
    r = http_json(f"https://itunes.apple.com/lookup?id={dev_id}&entity=software&limit=200")
    time.sleep(1.2)
    return [x for x in r.get("results", []) if x.get("kind") == "software"]


def hints(term):
    """App Store search autocomplete, in Apple's order."""
    url = ("https://search.itunes.apple.com/WebObjects/MZSearchHints.woa/wa/hints?clientApplication=Software&term="
           + urllib.parse.quote(term))
    raw = http_get(url, {"X-Apple-Store-Front": "143441-1,29"})
    time.sleep(0.35)
    try:
        return [h["term"].lower() for h in plistlib.loads(raw).get("hints", [])]
    except Exception:
        return []


def similar_ids(app_url):
    """Apple's 'You Might Also Like' shelf (customers-also-bought graph)."""
    h = http_get(app_url).decode("utf-8", "ignore")
    time.sleep(1.0)
    i = h.find('id="similarItems"')
    if i < 0:
        return []
    j = h.find('<section id="', i + 30)
    seg = h[i:j if j > 0 else i + 200000]
    return [int(x) for x in dict.fromkeys(re.findall(r'apps\.apple\.com/us/app/[^"/]+/id(\d+)', seg))]


def slim(x, source):
    return {
        "id": x["trackId"], "name": x.get("trackName", ""), "dev": x.get("artistName", ""),
        "devId": x.get("artistId"), "genre": x.get("primaryGenreName", ""),
        "desc": (x.get("description") or "")[:1500], "icon": x.get("artworkUrl512") or x.get("artworkUrl100"),
        "shots": (x.get("screenshotUrls") or [])[:4], "nShots": len(x.get("screenshotUrls") or []),
        "url": x.get("trackViewUrl", "").split("?")[0],
        "rating": round(x.get("averageUserRating") or 0, 1), "ratings": x.get("userRatingCount") or 0,
        "price": x.get("price") or 0, "released": (x.get("releaseDate") or "")[:10],
        "updated": (x.get("currentVersionReleaseDate") or "")[:10], "source": source,
    }


# ---- listing-quality signals -------------------------------------------------
def stale_days(a):
    try:
        return (TODAY - datetime.date.fromisoformat(a["updated"])).days
    except Exception:
        return 0


def niche_keyword(a):
    first = re.split(r"\s*[:\-–—|•(,]\s*", a["name"].lower())[0]
    words = [w for w in re.findall(r"[a-z0-9']+", first) if w not in NAME_STOP]
    return " ".join(words[:3]) or a["name"].lower()


# ---- text features ----------------------------------------------------------
def words_of(text):
    return [t for t in re.findall(r"[a-z][a-z']{2,}", text.lower()) if t not in STOP]


def tokens(a):
    w = words_of((a["name"] + " ") * 3 + a["desc"][:700])
    toks = set(w) | {w[i] + " " + w[i + 1] for i in range(len(w) - 1)}
    toks.add("genre:" + a["genre"].lower())
    r = a["ratings"]
    toks.add("pop:" + ("0" if r < 10 else "1" if r < 200 else "2" if r < 3000 else "3"))
    toks.add("price:" + ("free" if a["price"] == 0 else "paid"))
    return toks


def topic_words(a):
    return {w for w in re.findall(r"[a-z]{4,}", a["name"].lower()) if w not in NAME_STOP}


def seed_score(a):
    """Style prior from the reference portfolios: small, free, single-purpose utility apps."""
    name, desc = a["name"].lower(), a["desc"].lower()
    s = 0.35 if a["genre"].lower() in UTIL_GENRES else 0
    s += 0.2 if a["ratings"] < 3000 else 0
    s += 0.1 if a["price"] == 0 else 0
    s += min(0.35, 0.12 * sum(k in name for k in UTIL_KW) + 0.03 * sum(k in desc for k in UTIL_KW[:12]))
    return min(1.0, s)


# ---- semantic vectors (macOS NLEmbedding) ------------------------------------
WV = {"words": {}, "M": None}
DOC = {"key": None}


def ensure_word_vectors(vocab):
    if WV["M"] is None and os.path.exists(DP("wordvec.tsv")):
        _read_wordvec(DP("wordvec.tsv"))
    missing = [w for w in vocab if w not in WV["words"] and w not in WV.setdefault("none", set())]
    if not missing:
        return
    if sys.platform != "darwin":
        # NLEmbedding is macOS-only. In the cloud we use the cached vectors (wordvec.tsv, built on the Mac);
        # unseen words are skipped, the remaining words of each app still place it correctly.
        WV["none"] |= set(missing)
        return
    binp = DP("embed_bin")
    if not os.path.exists(binp):
        progress("Anlam modeli derleniyor (ilk sefer, ~30 sn)…")
        subprocess.run(["swiftc", "-O", P("embed.swift"), "-o", binp], check=True, capture_output=True)
    open(DP("words_in.txt"), "w").write("\n".join(missing))
    subprocess.run([binp, DP("words_in.txt"), DP("words_out.tsv")], check=True, capture_output=True)
    got = _read_wordvec(DP("words_out.tsv"))
    with open(DP("wordvec.tsv"), "a") as f:
        f.write(open(DP("words_out.tsv")).read())
    WV["none"] |= set(missing) - got


def _read_wordvec(path):
    words, vecs = [], []
    for line in open(path):
        w, v = line.rstrip("\n").split("\t")
        if w not in WV["words"]:
            words.append(w); vecs.append(np.array(v.split(","), dtype=np.float32))
    if words:
        base = 0 if WV["M"] is None else len(WV["M"])
        for k, w in enumerate(words):
            WV["words"][w] = base + k
        WV["M"] = np.vstack(([WV["M"]] if WV["M"] is not None else []) + [np.vstack(vecs)])
    return set(words)


def doc_vectors(apps):
    """IDF-weighted average word vectors (name words x3), mean-centred and L2-normalised. Cached."""
    key = (len(apps), apps[-1]["id"] if apps else 0)
    if DOC["key"] == key:
        return DOC
    docs = [(Counter(words_of(a["name"])), Counter(words_of(a["desc"])[:120])) for a in apps]
    df = Counter()
    for n, d in docs:
        df.update(set(n) | set(d))
    ensure_word_vectors(list({w for w in df if df[w] >= 2} | {w for n, _ in docs for w in n}))
    N, dim = len(apps), WV["M"].shape[1]
    D = np.zeros((N, dim), dtype=np.float32)
    for i, (n, d) in enumerate(docs):
        tot = 0.0
        for c, wt in ((n, 3.0), (d, 1.0)):
            for w, k in c.items():
                j = WV["words"].get(w)
                if j is None:
                    continue
                g = wt * min(k, 3) * math.log(N / df[w])
                D[i] += g * WV["M"][j]; tot += g
        if tot:
            D[i] /= tot
    has = np.abs(D).sum(1) > 0
    D[has] -= D[has].mean(0)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    # niche clusters: greedy, biggest apps become centres; everything within CLUSTER_SIM joins them
    nbrs = []
    for b in range(0, N, 512):
        S = D[b:b + 512] @ D.T
        nbrs += [np.nonzero(row > CLUSTER_SIM)[0] for row in S]
    cl = np.full(N, -1)
    for c in sorted(range(N), key=lambda i: -apps[i]["ratings"]):
        if cl[c] < 0:
            nb = nbrs[c]
            cl[nb[cl[nb] < 0]] = c
            cl[c] = c
    size = Counter(cl.tolist())
    DOC.update(key=key, D=D, idx={a["id"]: i for i, a in enumerate(apps)},
               cluster=cl, csize=np.array([size[c] for c in cl]))
    return DOC


# ---- model ------------------------------------------------------------------
def neg_weight(notes, i):
    rs = notes.get(str(i), {}).get("reasons") or []
    return max(REASON_W.get(r, 1.0) for r in rs) if rs else 1.0


def nb_weights(apps_by_id, pos, neg):
    pc, nc = Counter(), Counter()
    sp = sum(w for _, w in pos); sn = sum(w for _, w in neg)
    for i, w in pos:
        for t in tokens(apps_by_id[i]): pc[t] += w
    for i, w in neg:
        for t in tokens(apps_by_id[i]): nc[t] += w
    out = {}
    for t in set(pc) | set(nc):
        if pc[t] + nc[t] < 1.5 and not t.startswith(("genre:", "pop:", "price:")):
            continue
        out[t] = math.log((pc[t] + 0.5) / (sp + 1)) - math.log((nc[t] + 0.5) / (sn + 1))
    return out


def topk_mean(S, k=3):
    if S.shape[1] == 0:
        return np.zeros(S.shape[0], dtype=np.float32)
    k = min(k, S.shape[1])
    return np.sort(S, axis=1)[:, -k:].mean(1)


# Jev app features: [audience, pay, build, risk, kind_utility, kind_reference, kind_content, kind_game,
# kind_ai_wrapper, kind_service]. Prior = the brief (paid, buildable, niche utilities); labels pull it toward you.
JEV_PRIOR = np.array([0.8, 1.2, 0.8, -1.5, 0.6, 0.5, -0.6, -0.8, -0.6, -0.8], dtype=np.float64)
JEV_BIAS = -1.0
FEATURE_TR = ["meslek nişi", "ödeme isteği", "yapılabilirlik", "ret riski", "utility araç", "referans/eğitim",
              "içerik/medya", "oyun", "AI sohbet", "hizmet app'i"]
PREFS = {"weights": None, "n": 0}


def fit_jev_weights(X, y, sw):
    """Logistic regression, MAP with a Gaussian prior centred on JEV_PRIOR: few labels -> stays near the brief."""
    w, b = JEV_PRIOR.copy(), JEV_BIAS
    lam = 1.0
    for _ in range(600):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = (p - y) * sw
        w -= 0.3 * (X.T @ g / sw.sum() + lam * (w - JEV_PRIOR) / max(1.0, sw.sum()))
        b -= 0.3 * g.sum() / sw.sum()
    return w, b


def score_all(apps, labels, notes):
    by_id = {a["id"]: a for a in apps}
    dv = doc_vectors(apps)
    D, idx = dv["D"], dv["idx"]
    good = [int(i) for i, l in labels.items() if l == "good" and int(i) in by_id]
    bad = [(int(i), neg_weight(notes, i)) for i, l in labels.items() if l == "bad" and int(i) in by_id]
    bad = [(i, w) for i, w in bad if w > 0]
    refs = [a["id"] for a in apps if a["source"] == "ref" and str(a["id"]) not in labels]
    ref_w = 0.5 if len(good) < 3 else 0.2   # reference portfolios = your taste until you have labels
    pos = [(i, 1.0) for i in good] + [(i, ref_w) for i in refs]
    pos_rows = [idx[i] for i, _ in pos]
    pw = np.array([w for _, w in pos], dtype=np.float32)
    neg_rows = [idx[i] for i, _ in bad]
    nw = np.array([w for _, w in bad], dtype=np.float32)

    Sp = (D @ D[pos_rows].T) * pw if pos_rows else np.zeros((len(apps), 0), np.float32)
    Sn = (D @ D[neg_rows].T) * nw if neg_rows else np.zeros((len(apps), 0), np.float32)
    sp, sn = topk_mean(Sp), topk_mean(Sn)
    emb = 1 / (1 + np.exp(-(sp - sn) * 9)) if neg_rows else 1 / (1 + np.exp(-(sp - 0.25) * 9))
    Sg = D @ D[[idx[i] for i in good]].T if good else None   # unweighted, for clone detection

    use_nb = len(good) >= 2 and len(bad) >= 2
    nbw = nb_weights(by_id, [(i, 1.0) for i in good], bad) if use_nb else {}
    n_lab = len(good) + len(bad)
    trust = min(0.85, 0.45 + 0.02 * n_lab)
    kws = load("keywords.json", {})
    ads = load("ads.json", {})
    # how much you care about competition grows with how often you reject for "Çok rekabet"
    crowd_k = 0.1 * (1 + 3 * sum("Çok rekabet / doygun" in (n.get("reasons") or []) for n in notes.values())
                     / max(1, len(bad)))
    good_topics = [(by_id[i]["name"], topic_words(by_id[i])) for i in good]

    ja = load("jev_apps.json", {})
    Xl, yl, wl = [], [], []
    for i in good:
        if str(i) in ja:
            Xl.append(jev.app_vector(ja[str(i)])); yl.append(1.0); wl.append(1.0)
    for i, w in bad:
        if str(i) in ja:
            Xl.append(jev.app_vector(ja[str(i)])); yl.append(0.0); wl.append(w)
    if len(Xl) >= 4 and 0 < sum(yl) < len(yl):
        jw, jb = fit_jev_weights(np.array(Xl), np.array(yl), np.array(wl))
    else:
        jw, jb = JEV_PRIOR, JEV_BIAS
    PREFS.update(weights=[{"t": FEATURE_TR[k], "w": round(float(jw[k]), 2)} for k in np.argsort(-np.abs(jw))],
                 n=len(Xl))

    out = {}
    for r, a in enumerate(apps):
        why = []
        learned = float(emb[r])
        if use_nb:
            contrib = sorted(((nbw[k], k) for k in tokens(a) if k in nbw), key=lambda z: -abs(z[0]))
            s = sum(c for c, _ in contrib[:40]) + math.log((len(good) + 1) / (len(bad) + 1))
            learned = 0.65 * learned + 0.35 / (1 + math.exp(-s / 6))
            why += [{"t": k, "w": round(c, 2)} for c, k in contrib[:3] if abs(c) > 0.9 and ":" not in k]
        badges = []
        jf = ja.get(str(a["id"]))
        if jf:
            jp = float(1 / (1 + np.exp(-(np.array(jev.app_vector(jf)) @ jw + jb))))
            learned = 0.55 * learned + 0.45 * jp
            badges.append({"t": f"Jev: meslek {jf['audience']:.1f} · ödeme {jf['pay']:.1f} · yapılır {jf['build']:.1f}",
                           "k": "ok" if jp > 0.6 else "bad" if jp < 0.3 else ""})
            if jf["risk"] >= 0.5:
                badges.append({"t": f"Ret riski {jf['risk']:.0%}", "k": "bad"})
        score = trust * learned + (1 - trust) * seed_score(a)
        if Sp.shape[1]:
            j = int(np.argmax(Sp[r]))
            near = by_id[pos[j][0]]
            if near["id"] != a["id"]:
                why.insert(0, {"t": "benzer: " + near["name"][:28], "w": 1})
        if neg_rows and float(sn[r]) > 0.3:
            j = int(np.argmax(Sn[r]))
            why.insert(1, {"t": "kötüye benzer: " + by_id[bad[j][0]]["name"][:24], "w": -1})
        # clone of something you already liked -> not a new idea
        if Sg is not None and str(a["id"]) not in labels:
            j = int(np.argmax(Sg[r]))
            tw = topic_words(a)
            jac = max((len(tw & t) / max(1, len(tw | t)) for _, t in good_topics), default=0)
            if Sg[r, j] > CLONE_SIM or jac >= 0.5:
                score -= 0.2
                badges.append({"t": "Klon: " + by_id[good[j]]["name"][:22], "k": "bad"})
        # niche opportunity, measured on the live App Store top-10 (Niş Bulucu)
        nk = niche_keyword(a)
        kw = kws.get(nk)
        if kw:
            kw = {**kw, "ads": ads.get(nk)}
            score += crowd_k * (kw["weak"] - 50) / 50 + 0.05 * (kw["pop"] - 30) / 70
            score += 0.05 if kw.get("ads") == "no" else -0.03 if kw.get("ads") == "yes" else 0
            badges.append({"t": f"Niş fırsatı {kw['opp']} · arama {kw['pop']} · rakip zayıflığı {kw['weak']}",
                           "k": "ok" if kw["opp"] >= 45 else "bad" if kw["opp"] < 25 else ""})
        # weak listing = opportunity in this niche
        sd = stale_days(a)
        if sd > 365:
            badges.append({"t": f"{sd / 365:.1f} yıldır güncelleme yok", "k": "ok"})
        if a.get("nShots", 4) and a.get("nShots", 4) < 3:
            badges.append({"t": "Az ekran görüntüsü", "k": "ok"})
        if a["rating"] and a["rating"] < 4 and a["ratings"] >= 20:
            badges.append({"t": f"★{a['rating']}: kullanıcı mutsuz", "k": "ok"})
        cs = int(dv["csize"][r])
        if cs >= 5:
            badges.append({"t": f"Listede bu nişten {cs} app var", "k": ""})
        out[a["id"]] = {"score": max(0.0, min(1.0, score)), "why": why[:5], "badges": badges,
                        "niche": niche_keyword(a), "cluster": int(dv["cluster"][r])}
    return out, len(good), len(bad)


def fmt(n):
    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)


def diversify(apps, top=600):
    """Greedy re-rank: each further app from an already-shown niche cluster (or name topic) loses rank."""
    head, tail = apps[:top], apps[top:]
    seen, seen_c, out = Counter(), Counter(), []
    tw = {a["id"]: topic_words(a) for a in head}
    while head:
        best = max(head, key=lambda a: a["rank"] - 0.25 * seen_c[a["cluster"]]
                   - 0.15 * max([seen[w] for w in tw[a["id"]]] or [0]))
        head.remove(best); out.append(best); seen.update(tw[best["id"]]); seen_c[best["cluster"]] += 1
    return out + tail


# ---- discovery job ----------------------------------------------------------
def discover_job():
    with LOCK:
        apps, labels, notes = load("candidates.json", []), load("labels.json", {}), load("notes.json", {})
    by_id = {a["id"]: a for a in apps}
    liked = [by_id[int(i)] for i, l in labels.items() if l == "good" and int(i) in by_id][-10:]
    seeds = liked or random.sample([a for a in apps if a["source"] == "ref"], 6)
    found = {}   # id -> source

    # 1) Apple's "You Might Also Like" graph: same audience, different apps
    for k, a in enumerate(seeds):
        progress(f"Apple benzerleri {k + 1}/{len(seeds)}: {a['name']}")
        for i in similar_ids(a["url"]):
            found.setdefault(i, "similar")

    # 2) autocomplete around each liked niche + strongest learned phrases
    queries = []
    for a in seeds[:6]:
        kw = niche_keyword(a)
        queries += [h for h in hints(kw) if h != kw][:2]
    if len(liked) >= 2 and labels:
        bad = [(int(i), neg_weight(notes, i)) for i, l in labels.items() if l == "bad" and int(i) in by_id]
        nbw = nb_weights(by_id, [(a["id"], 1.0) for a in liked], [(i, w) for i, w in bad if w > 0])
        queries += [t for t, w in sorted(nbw.items(), key=lambda z: -z[1]) if " " in t and ":" not in t][:6]
    queries = list(dict.fromkeys(queries))[:18]
    for k, q in enumerate(queries):
        progress(f"Arama {k + 1}/{len(queries)}: {q}")
        for x in search(q, 25):
            found.setdefault(x["trackId"], "discovered")

    # 3) portfolios of small (indie) developers you liked
    raw = {}
    for d in list(dict.fromkeys(a["devId"] for a in liked if a.get("devId")))[:6]:
        progress("Developer portföyü taranıyor…")
        r = dev_apps(d)
        if len(r) <= 30:
            for x in r:
                raw[x["trackId"]] = x; found.setdefault(x["trackId"], "discovered")

    new_ids = [i for i in found if i not in by_id and i not in raw]
    progress(f"{len(new_ids)} app detayı çekiliyor…")
    for x in lookup(new_ids):
        raw[x["trackId"]] = x
    with LOCK:
        cands = load("candidates.json", [])
        known = {a["id"] for a in cands}
        added = [slim(x, found.get(i, "discovered")) for i, x in raw.items() if i not in known]
        cands += added
        save("candidates.json", cands)
    progress("Anlam vektörleri güncelleniyor…")
    doc_vectors(cands)
    return {"added": len(added), "similar": sum(a["source"] == "similar" for a in added), "queries": queries}


# ---- niche finder job -------------------------------------------------------
def popularity(term):
    """Autocomplete prefix test: the fewer letters needed before Apple suggests the term, the more it is searched."""
    lens = sorted({L for L in (2, 3, 4, 5, 7, 9, 12, len(term)) if 2 <= L <= len(term)})
    for L in lens:
        hs = hints(term[:L])
        # exact match, or Apple suggests a longer form of it ("breathing exercise" -> "breathing exercises")
        pos = next((i for i, h in enumerate(hs) if h == term or h.startswith(term)), None)
        if pos is not None:
            return pop_score(L, pos, hs[pos] != term), L, pos
    return 0, None, None


def pop_score(L, pos=0, partial=False):
    """Absolute letters typed before Apple suggests it: 2 letters = 100, each extra letter -9, lower in list -3."""
    if L is None:
        return 0
    return max(5, round(100 - 9 * (L - 2) - 3 * (pos or 0) - (5 if partial else 0)))


TERM_STOP = {"for", "app", "the", "and", "of", "to", "a", "in", "with", "free", "my", "&"}


def term_in_name(term, name):
    """All content words of the term appear in the app name (tolerates plural/'-er' forms)."""
    name = name.lower()
    ws = [w for w in re.findall(r"[a-z0-9]+", term.lower()) if w not in TERM_STOP]
    return bool(ws) and all(w in name or (len(w) > 4 and w[:-2] in name) for w in ws)


# a term with one of these words describes a *need* (tool/knowledge), not one company's app
INTENT = set("calculator calc calculation calculations guide chart charts symbols symbol test tests exam exams prep quiz study "
             "practice flashcards reference log logbook tracker simulator game timer tool tools codes notes checklist "
             "map finder scanner journal manual handbook formulas formula conversion converter estimator planner "
             "atlas anatomy protocols terminology dictionary settings parameters inspection training learn learning "
             "points meter app review board boards drills exercises workout scheduler sizing table tables measure pre trip".split())
BUSINESS = set("llc inc pc pa ltd office clinic clinics studio association federal credit union center centre "
               "group family associates partners church hospital".split())


def rival_stats(term, top):
    """Rivals = top-10 apps that carry the term in their title (= really target this niche).
    brand = looks like a search for one specific business/app rather than for a kind of tool."""
    riv = [a for a in top if term_in_name(term, a["name"])]
    ws = set(re.findall(r"[a-z0-9]+", term.lower()))
    brand = bool(ws & BUSINESS) or (len(riv) <= 1 and not (ws & INTENT) and len(ws) > 1)
    return {"rivals": len(riv), "serious": sum(a["ratings"] >= 50 for a in riv), "brand": brand}


def analyze_term(term, top=None):
    pop, at, hpos = popularity(term)
    tails = [h for h in hints(term) if h != term][:8]
    if top is None:
        top = [slim(x, "niche") for x in search(term, 10)]
    rs = [a["ratings"] for a in top] or [0]
    med = statistics.median(rs)
    n = max(1, len(top))
    stale = sum(stale_days(a) > 365 for a in top) / n
    tw = term.split()
    title = sum(all(w in a["name"].lower() for w in tw) for a in top) / n
    rated = [a["rating"] for a in top if a["ratings"] >= 5]
    avg = sum(rated) / len(rated) if rated else 4.5
    few_shots = sum(a.get("nShots", 5) < 4 for a in top) / n
    weak = (35 * (1 - min(1, math.log10(med + 1) / 4)) + 25 * stale + 15 * (1 - title)
            + 15 * min(1, max(0, (4.6 - avg) / 1.2)) + 10 * few_shots)
    return {"term": term, "pop": pop, "foundAt": at, "hintPos": hpos, "tails": tails, "median": int(med), "max": max(rs),
            **rival_stats(term, top),
            "stale": round(stale, 2), "fewShots": round(few_shots, 2), "title": round(title, 2), "avg": round(avg, 2),
            "weak": round(weak), "opp": round(math.sqrt(max(pop, 0) * weak)), "date": str(TODAY),
            "top": [{k: a[k] for k in ("id", "name", "icon", "ratings", "rating", "updated", "url", "shots", "dev")}
                    | {"desc": a["desc"][:300], "stale": stale_days(a) > 365, "nShots": a["nShots"]} for a in top]}


def niche_job(terms, auto):
    if auto:
        with LOCK:
            apps, labels = load("candidates.json", []), load("labels.json", {})
        by_id = {a["id"]: a for a in apps}
        liked = [by_id[int(i)] for i, l in labels.items() if l == "good" and int(i) in by_id]
        terms = [niche_keyword(a) for a in liked][-12:] + IDEA_TERMS
        progress("Beğendiklerinin uzun kuyruk kelimeleri toplanıyor…")
        for t in [niche_keyword(a) for a in liked][-6:]:
            terms += [h for h in hints(t) if h != t][:2]
    kws = load("keywords.json", {})
    terms = [t.strip().lower() for t in dict.fromkeys(terms) if t.strip()]
    if auto:
        terms = [t for t in terms if t not in kws]
    for k, t in enumerate(terms):
        progress(f"Niş analizi {k + 1}/{len(terms)}: {t}")
        res = analyze_term(t)
        add_jev_keyword(res)
        with LOCK:
            kws = load("keywords.json", {})
            kws[t] = res
            save("keywords.json", kws)
    return {"analyzed": len(terms)}


PROFESSIONS = ["chiropractic", "physical therapy", "occupational therapy", "speech therapy", "sonography",
               "phlebotomy", "dental hygiene", "dental assistant", "pharmacy technician", "paramedic",
               "respiratory therapy", "midwife", "optometry", "audiology", "vet tech", "massage therapy",
               "acupuncture", "dietitian", "personal trainer", "pilates", "pilot", "electrician", "plumber",
               "hvac", "welding", "carpentry", "machinist", "surveying", "forklift", "trucker", "firefighter",
               "notary", "paralegal", "court interpreter", "tattoo", "barber", "esthetician", "nail tech",
               "florist", "beekeeping", "farrier", "dog grooming", "bartender", "sommelier", "piano tuning",
               "gemology", "geology", "birding", "sailing", "scuba"]


def mine_job(seeds):
    """For each profession: mine App Store autocomplete (seed + a..z), count title rivals per term,
    then run the full niche analysis only on terms with <= 2 rivals."""
    kws = load("keywords.json", {})
    save("mine_queue.json", seeds)   # survives restarts: the server resumes the remaining professions
    for k, seed in enumerate(seeds):
        seed = seed.strip().lower()
        progress(f"Meslek {k + 1}/{len(seeds)}: {seed} — otomatik tamamlama taranıyor")
        rank = {}
        for suffix in [""] + [" " + c for c in "abcdefghijklmnoprstuvwy"]:
            for i, h in enumerate(hints(seed + suffix)):
                rank[h] = min(rank.get(h, 99), i)
        # keep terms Apple ranks high, skip ones already analysed
        terms = [t for t, _ in sorted(rank.items(), key=lambda z: z[1]) if t not in kws][:14]
        for j, t in enumerate(terms):
            progress(f"Meslek {k + 1}/{len(seeds)}: {seed} — rakip sayımı {j + 1}/{len(terms)}: {t}")
            top = [slim(x, "niche") for x in search(t, 10)]
            if rival_stats(t, top)["rivals"] > 2:
                continue
            progress(f"Meslek {k + 1}/{len(seeds)}: {seed} — popülerlik: {t}")
            res = analyze_term(t, top)
            res["seed"] = seed
            add_jev_keyword(res)
            with LOCK:
                kws = load("keywords.json", {})
                kws[t] = res
                save("keywords.json", kws)
        save("mine_queue.json", seeds[k + 1:])
    return {"analyzed": len(seeds), "mined": True}


# ---- Jev (TypeSafe) enrichment --------------------------------------------------
JEVJOB = {"running": False, "progress": "", "result": None}


def add_jev_keyword(k):
    if not jev.api_key():
        return
    try:
        k["jev"] = jev.judge_keyword(k["term"], k.get("seed"), k["top"])
    except Exception as e:
        k["jevError"] = str(e)[:200]


def jev_keywords_job(force=False):
    kws = load("keywords.json", {})
    todo = [k for k in kws.values() if force or "jev" not in k]
    # older rows lack developer/description for the top-10: fetch them once
    need = {a["id"] for k in todo for a in k["top"] if "desc" not in a}
    if need:
        JEVJOB["progress"] = f"Rakip açıklamaları çekiliyor ({len(need)} app)…"
        full = {x["trackId"]: slim(x, "niche") for x in lookup(need)}
        for k in todo:
            for a in k["top"]:
                if "desc" not in a and a["id"] in full:
                    a["desc"], a["dev"] = full[a["id"]]["desc"][:300], full[a["id"]]["dev"]
    done = 0

    def one(k):
        add_jev_keyword(k)
        return k

    with ThreadPoolExecutor(6) as ex:
        for k in ex.map(one, todo):
            done += 1
            JEVJOB["progress"] = f"Jev nişleri değerlendiriyor {done}/{len(todo)}"
            if done % 20 == 0 or done == len(todo):
                _merge_keywords(todo[:done])
    return {"keywords": len(todo)}


def _merge_keywords(rows):
    with LOCK:
        kws = load("keywords.json", {})
        for k in rows:
            if k["term"] in kws and "jev" in k:
                kws[k["term"]]["jev"] = k["jev"]
                kws[k["term"]]["top"] = k["top"]
        save("keywords.json", kws)


def jev_apps_job(force=False):
    """Judge every app once (kind, audience, willingness to pay, buildability, review risk).
    Labelled apps first, then the rest in current rank order."""
    with LOCK:
        apps, labels = load("candidates.json", []), load("labels.json", {})
    have = {} if force else load("jev_apps.json", {})
    todo = [a for a in apps if str(a["id"]) not in have]
    todo.sort(key=lambda a: (str(a["id"]) not in labels, -seed_score(a)))
    done, batch = 0, {}

    def one(a):
        try:
            return a["id"], jev.judge_app(a)
        except Exception as e:
            return a["id"], None

    with ThreadPoolExecutor(8) as ex:
        for i, j in ex.map(one, todo):
            done += 1
            if j:
                batch[str(i)] = j
            JEVJOB["progress"] = f"Jev app'leri değerlendiriyor {done}/{len(todo)}"
            if len(batch) >= 100 or done == len(todo):
                with LOCK:
                    cur = load("jev_apps.json", {}); cur.update(batch); save("jev_apps.json", cur)
                batch = {}
    return {"apps": done}


def jev_all_job(force=False):
    if not jev.api_key():
        return {"error": "TYPESAFE_API_KEY yok (.env)"}
    r = jev_keywords_job(force)
    r.update(jev_apps_job(force))
    return r


def start_jev_job(force=False):
    if JEVJOB["running"]:
        return False

    def run():
        try:
            JEVJOB["result"] = jev_all_job(force)
        except Exception as e:
            JEVJOB["result"] = {"error": str(e)}
        JEVJOB.update(running=False, progress="")
    JEVJOB.update(running=True, progress="Başlıyor…", result=None)
    threading.Thread(target=run, daemon=True).start()
    return True


def note_to_reason(i, note):
    """Free-text 'bad' note without a picked reason -> map it to a reason with Jev (kept only if confident)."""
    try:
        r, conf = jev.note_reason(note, list(REASON_W))
    except Exception:
        return
    if r in REASON_W and conf >= 0.5:
        with LOCK:
            notes = load("notes.json", {})
            if str(i) in notes and not notes[str(i)].get("reasons"):
                notes[str(i)]["reasons"] = [r]
                notes[str(i)]["auto"] = True
                save("notes.json", notes)


def start_job(kind, fn, *args):
    if JOB["running"]:
        return False

    def run():
        try:
            JOB["result"] = fn(*args)
        except Exception as e:
            JOB["result"] = {"error": str(e)}
        JOB["running"] = False
        JOB["progress"] = ""
    JOB.update(running=True, kind=kind, progress="Başlıyor…", result=None)
    threading.Thread(target=run, daemon=True).start()
    return True


# ---- GitHub mode ---------------------------------------------------------------
# Heavy jobs run in GitHub Actions (.github/workflows/scout.yml) on a schedule, even when the Mac is off.
# The Mac runs only the labelling UI and syncs: it pushes MAC_FILES, Actions pushes the rest -> no conflicts.
GITHUB_MODE = os.environ.get("SCOUT_GITHUB") == "1"
MAC_FILES = ["labels.json", "notes.json", "ads.json", "hidden.json", "wordvec.tsv"]
GH = {"run": "", "last_sync": ""}


def git(*args, timeout=120):
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, timeout=timeout)


def gh_dispatch(task, arg=None):
    r = subprocess.run(["gh", "workflow", "run", "scout.yml", "-f", f"task={task}",
                        "-f", "arg=" + (json.dumps(arg, ensure_ascii=False) if arg else "")],
                       cwd=HERE, capture_output=True, text=True, timeout=60)
    return r.returncode == 0, (r.stderr or r.stdout).strip()[:300]


def sync_once():
    with LOCK:
        files = [f for f in MAC_FILES if os.path.exists(P(f))]
        git("add", *files)
        if git("diff", "--cached", "--quiet").returncode != 0:
            git("commit", "-q", "-m", "scout: etiketler (Mac)")
        pull = git("pull", "--rebase", "--autostash", "-q")
        push = git("push", "-q")
    GH["last_sync"] = datetime.datetime.now().strftime("%H:%M")
    if pull.returncode or push.returncode:
        GH["last_sync"] += " (hata: " + (pull.stderr or push.stderr).strip()[:120] + ")"


def gh_status():
    r = subprocess.run(["gh", "run", "list", "--workflow", "scout.yml", "--limit", "1", "--json",
                        "status,conclusion,displayTitle,createdAt"], cwd=HERE, capture_output=True, text=True, timeout=60)
    try:
        x = json.loads(r.stdout)[0]
        state = {"in_progress": "çalışıyor", "queued": "sırada", "completed": x.get("conclusion") or "bitti"}
        GH["run"] = f"GitHub: {x['displayTitle']} · {state.get(x['status'], x['status'])}"
    except Exception:
        GH["run"] = ""


def sync_loop():
    n = 0
    while True:
        try:
            gh_status()
            if n % 3 == 0:
                sync_once()
        except Exception as e:
            GH["last_sync"] = "hata: " + str(e)[:100]
        n += 1
        time.sleep(60)


def run_task(task, arg):
    """Synchronous entry point used by GitHub Actions (python server.py --task ...)."""
    if task == "discover":
        return discover_job()
    if task == "mine":
        return mine_job(arg or load("mine_queue.json", []) or PROFESSIONS)
    if task == "niche":
        return niche_job(arg or [], not arg)
    if task == "jev":
        return jev_all_job()
    if task == "rejudge":
        return jev_all_job(force=True)
    if task == "autopilot":
        out = {}
        if load("mine_queue.json", []):
            out["mine"] = mine_job(load("mine_queue.json", []))
        out["discover"] = discover_job()
        out["niche"] = niche_job([], True)
        out["jev"] = jev_all_job()
        save("autopilot.json", {"last": datetime.datetime.now().isoformat(timespec="minutes"), "result": out})
        return out
    raise SystemExit(f"bilinmeyen görev: {task}")


# ---- autopilot (cloud): keeps finding, analysing and judging while you are away ----------
def autopilot(hours):
    """Every `hours`: Apple-similar discovery -> niche analysis of new liked niches -> Jev judgments.
    Each step waits for a free job slot, so manual runs from the UI are never interrupted."""
    def wait_free():
        while JOB["running"]:
            time.sleep(30)

    while True:
        try:
            q = load("mine_queue.json", [])
            if q:
                wait_free(); start_job("niche", mine_job, q)
            wait_free(); start_job("discover", discover_job)
            wait_free(); start_job("niche", niche_job, [], True)
            wait_free()
            if jev.api_key():
                while JEVJOB["running"]:
                    time.sleep(30)
                start_jev_job()
            save("autopilot.json", {"last": datetime.datetime.now().isoformat(timespec="minutes"), "hours": hours})
        except Exception as e:
            print("autopilot error:", e, flush=True)
        time.sleep(hours * 3600)


def seed_data_dir():
    """First start on a fresh cloud volume: copy the bundled data so labels/analyses carry over."""
    if DATA_DIR == HERE:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in ("candidates.json", "labels.json", "notes.json", "keywords.json", "jev_apps.json",
              "wordvec.tsv", "mine_queue.json"):
        if os.path.exists(P(f)) and not os.path.exists(DP(f)):
            shutil.copy(P(f), DP(f))


# ---- HTTP ------------------------------------------------------------------
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, code=200, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authed(self):
        pw = os.environ.get("APP_PASSWORD")
        if not pw:
            return True
        h = self.headers.get("Authorization", "")
        try:
            ok = h.startswith("Basic ") and base64.b64decode(h[6:]).decode().split(":", 1)[1] == pw
        except Exception:
            ok = False
        if not ok:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="App Scout"')
            self.send_header("Content-Length", "0")
            self.end_headers()
        return ok

    def do_GET(self):
        if self.path == "/healthz":
            return self.send({"ok": True})
        if not self.authed():
            return
        if self.path in ("/", "/index.html"):
            return self.send(open(P("index.html"), "rb").read(), ctype="text/html")
        if self.path == "/api/status":
            return self.send({**JOB, "jev": JEVJOB, "jevKey": bool(jev.api_key()) or GITHUB_MODE,
                              "github": GITHUB_MODE, "gh": GH})
        if self.path == "/api/niche":
            ads, hidden = load("ads.json", {}), set(load("hidden.json", []))
            kws = [k for k in load("keywords.json", {}).values() if k["term"] not in hidden]
            for k in kws:
                k["ads"] = ads.get(k["term"])
                k.update(rival_stats(k["term"], k["top"]))   # recomputed so rule changes apply to old rows
                k["pop"] = pop_score(k.get("foundAt"), k.get("hintPos"))
                k["opp"] = round(math.sqrt(k["pop"] * k["weak"]))
                j = k.get("jev")
                if j:   # Jev: real rivals = top-10 apps that actually serve this need
                    real = [a for a, p in zip(k["top"], j["rivalP"]) if p >= 0.5]
                    k["rivals"], k["serious"] = len(real), sum(a["ratings"] >= 50 for a in real)
                    k["brand"] = j["intent"] in ("brand", "local_business")
                    k["intent"] = j["intent"]
                    k["quality"] = round(100 * (0.35 * j["audience"] + 0.4 * j["pay"] + 0.25 * j["build"]))
                bonus = (12 if k["rivals"] <= 1 else 5 if k["rivals"] == 2 else 0)
                if j:
                    bonus += round((k["quality"] - 50) / 4) - round(20 * j["risk"])
                k["score"] = max(0, min(100, k["opp"] + bonus
                                        + (10 if k.get("ads") == "no" else -10 if k.get("ads") == "yes" else 0)))
                k["gold"] = (k["rivals"] <= 1 and k["pop"] >= 30 and k["weak"] >= 35 and not k["brand"]
                             and (not j or (j["intent"] == "need" and j["risk"] < 0.5)))
            return self.send(sorted(kws, key=lambda k: -k["score"]))
        if self.path == "/api/apps":
            with LOCK:
                apps, labels, notes = load("candidates.json", []), load("labels.json", {}), load("notes.json", {})
                sc, npos, nneg = score_all(apps, labels, notes)
            noise = 0.12 if npos + nneg >= 6 else 0.35
            for a in apps:
                a.update(sc[a["id"]]); a["label"] = labels.get(str(a["id"]))
                nt = notes.get(str(a["id"]), {})
                a["reasons"] = nt.get("reasons", []); a["note"] = nt.get("note", "")
                a["rank"] = a["score"] + random.random() * noise
            apps.sort(key=lambda a: -a["rank"])
            apps = diversify(apps)
            return self.send({"apps": apps, "good": npos, "bad": nneg, "prefs": PREFS})
        self.send({"error": "not found"}, 404)

    def do_POST(self):
        if not self.authed():
            return
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if GITHUB_MODE and self.path in ("/api/discover", "/api/mine", "/api/niche", "/api/jev"):
            task = {"/api/discover": "discover", "/api/mine": "mine", "/api/niche": "niche",
                    "/api/jev": "rejudge" if data.get("rejudge") else "jev"}[self.path]
            arg = data.get("seeds") if self.path == "/api/mine" else data.get("terms") if self.path == "/api/niche" else None
            ok, msg = gh_dispatch(task, arg or None)
            return self.send({"started": ok, "github": True, "message": msg})
        if self.path == "/api/label":
            with LOCK:
                labels, notes = load("labels.json", {}), load("notes.json", {})
                k = str(data["id"])
                labels.pop(k, None)   # re-insert so dict order = labelling order
                notes.pop(k, None)
                if data.get("label") in ("good", "bad"):
                    labels[k] = data["label"]
                    if data.get("reasons") or data.get("note"):
                        notes[k] = {"reasons": data.get("reasons", []), "note": data.get("note", ""),
                                    "label": data["label"], "name": data.get("name", "")}
                save("labels.json", labels); save("notes.json", notes)
            if data.get("label") == "bad" and data.get("note") and not data.get("reasons") and jev.api_key():
                threading.Thread(target=note_to_reason, args=(k, data["note"]), daemon=True).start()
            return self.send({"ok": True, "n": len(labels)})
        if self.path == "/api/discover":
            return self.send({"started": start_job("discover", discover_job)})
        if self.path == "/api/niche":
            return self.send({"started": start_job("niche", niche_job, data.get("terms", []), bool(data.get("auto")))})
        if self.path == "/api/jev":
            return self.send({"started": start_jev_job(bool(data.get("rejudge")))})
        if self.path == "/api/mine":
            seeds = [x for x in data.get("seeds", []) if x.strip()] or PROFESSIONS
            return self.send({"started": start_job("niche", mine_job, seeds)})
        if self.path == "/api/niche/ads":
            with LOCK:
                ads = load("ads.json", {})
                if data.get("ads"):
                    ads[data["term"]] = data["ads"]
                else:
                    ads.pop(data["term"], None)
                save("ads.json", ads)
            return self.send({"ok": True})
        if self.path == "/api/niche/delete":
            with LOCK:
                hidden = load("hidden.json", [])
                if data["term"] not in hidden:
                    hidden.append(data["term"]); save("hidden.json", hidden)
            return self.send({"ok": True})
        self.send({"error": "not found"}, 404)


if __name__ == "__main__":
    if "--task" in sys.argv:   # GitHub Actions: run one job and exit
        task = sys.argv[sys.argv.index("--task") + 1]
        raw = sys.argv[sys.argv.index("--arg") + 1] if "--arg" in sys.argv else ""
        arg = json.loads(raw) if raw.strip() else None
        print(json.dumps(run_task(task, arg), ensure_ascii=False)[:2000], flush=True)
        sys.exit(0)
    port = int(os.environ.get("PORT", 8765))
    host = os.environ.get("HOST", "127.0.0.1")   # cloud: 0.0.0.0
    seed_data_dir()
    print("Anlam vektörleri hazırlanıyor…", flush=True)
    doc_vectors(load("candidates.json", []))
    hours = float(os.environ.get("AUTOPILOT_HOURS", "0"))
    if hours > 0:
        threading.Thread(target=autopilot, args=(hours,), daemon=True).start()
        print(f"Autopilot: every {hours}h", flush=True)
    elif GITHUB_MODE:
        threading.Thread(target=sync_loop, daemon=True).start()
        print("GitHub modu: ağır işler GitHub Actions'ta, Mac sadece etiketliyor ve eşitliyor", flush=True)
    elif load("mine_queue.json", []):   # local: just finish an interrupted profession scan
        start_job("niche", mine_job, load("mine_queue.json", []))
    print(f"App Scout on http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), H).serve_forever()
