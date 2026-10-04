"""TypeSafe Jev client + the judgments App Scout asks it. API: https://docs.typesafe.ai/api.md

Jev returns typed judgments (choice / score / noul probabilities), not text. Code keeps the policy:
which answers count, thresholds and weights all live in server.py.

Questions are versioned in jev_registry.json. jev_optimize.py tries new versions every hour against the
hand-labelled bench (bench/data.json) and promotes one only by consensus (dev win + no test loss + repeat).
If a capability has several production versions, their answers are averaged (ensemble).
"""
import copy, json, os, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
REGISTRY = os.path.join(HERE, "jev_registry.json")


def api_key():
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    try:
        for line in open(os.path.join(HERE, ".env")):
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    return None


def ask(state, questions, tries=5):
    """One request: every question is evaluated in parallel against the same state."""
    body = json.dumps({"state": state, "model": MODEL, "questions": questions}).encode()
    for attempt in range(tries):
        req = urllib.request.Request(URL, data=body, headers={
            "Authorization": "Bearer " + api_key(), "Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise RuntimeError(f"Jev HTTP {e.code}: {e.read()[:300]!r}")
            time.sleep(float(e.headers.get("retry-after") or 2 * (attempt + 1)))
        except urllib.error.URLError:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def norm(ans):
    """Score -> 0..1 position on its levels; noul -> probability of yes; choice -> chosen option."""
    if ans["type"] == "noul":
        return round(ans["noul"], 3)
    if ans["type"] == "score":
        return round(ans["score"] / max(1, len(ans["legend"]) - 1), 3)
    return ans["choice"]


# ---- version-1 rubrics (also the fallback when the registry is missing) -----------
AUDIENCE = ["The general public; anyone could need it",
            "A broad hobby or interest group",
            "People in a specific occupation or trade, or students training for one",
            "A narrow specialty inside an occupation"]
PAY = ["Expects it to be free; would not pay",
       "Might pay a small one-time amount",
       "Would pay a recurring subscription because it helps their work, income or an exam they must pass"]
BUILD = ["Not feasible: needs licensed data, special hardware, partnerships, a big content team or a regulated service",
         "Hard: needs a lot of expert content or complex engineering",
         "Feasible: moderate content or engineering that one developer can produce or source",
         "Easy: mostly interface over public knowledge, formulas or the phone's own sensors"]
RISK = {"true": "The app itself would do something App Review restricts: calculate drug doses for real patients, "
                "diagnose or treat medical conditions, give regulated financial or legal advice, run real-money gambling, "
                "distribute copyrighted material, or operate a service that needs an official licence",
        "false": "An ordinary app. Study aids, exam prep, flashcards, reference guides, calculators for trades and "
                 "educational content ABOUT medicine, law or finance are ordinary apps"}
RISK_Q = "be restricted by App Store review because of what the app itself does"
INTENT = {
    "need": "A kind of tool, calculator, reference, study aid or utility; any good app doing this job would satisfy them",
    "brand": "One specific company, product, publisher or app, by its name",
    "local_business": "The app of one specific clinic, shop, school, credit union or other organization",
    "game": "A game or simulator to play for fun",
    "jobs": "Job listings, hiring or staffing",
    "other": "None of the above",
}
KIND = {
    "utility": "A tool that does one job with the phone's sensors, camera, audio, files or calculations",
    "reference": "Reference, learning, study or exam-prep content for a topic",
    "content": "Media, entertainment, streaming, social or news",
    "game": "A game",
    "ai_wrapper": "Mainly a chat or AI-generation interface",
    "service": "Front end for a business, account or online service",
}
REASON_DESC = {
    "Alakasız": "The app or idea is unrelated to what the user is looking for",
    "Çok rekabet / doygun": "The market niche already has strong, established competitors",
    "Yapması zor": "Too hard or expensive for the user to build",
    "Para yapmaz": "Users would not pay; it would not make money",
    "Kalitesiz / eski": "This particular app listing is low quality or outdated",
    "Çok benzer": "The user keeps seeing too many near-identical apps in this list; it is repetitive",
    "Ret riski": "Likely to be rejected by App Store review",
    "Sıkıcı fikir": "The idea is boring or uninteresting to the user",
}

# capability -> (question type, version-1 spec). Placeholders: {i} result index, {subject} what is judged.
TYPES = {"intent": "choice", "brand": "noul", "rival": "noul", "risk": "noul", "audience": "score",
         "pay": "score", "build": "score", "kind": "choice", "note": "choice"}
V1 = {
    "intent": {"instructions": "What is a person who types `search_term` into the App Store search box looking for?",
               "criteria": INTENT},
    "brand": {"instructions": "Is `search_term` a proper name - the brand, product, company or app name of one specific "
                              "app or company - rather than plain words describing what the searcher needs? "
                              "Use `top_results` as evidence: a brand usually appears as the distinctive title of one result.",
              "criteria": {"true": "A proper or brand name (like 'ForeFlight', 'PassRun', 'Smart Optometry'), possibly with generic words added",
                           "false": "Plain descriptive words for a need or topic, even if one app happens to use these words as its title "
                                    "(like 'paramedic exam prep', 'welding symbols', 'acupuncture tools')"}},
    "rival": {"instructions": "Does `top_results[{i}]` directly serve what a person searching `search_term` wants?",
              "criteria": {"true": "Built for this need: a real competitor",
                           "false": "Unrelated, only loosely related, or a general app that merely mentions it"}},
    "risk": {"instructions": "Would {subject} likely " + RISK_Q + "?", "criteria": RISK},
    "audience": {"instructions": "Who is {subject} built for?", "criteria": AUDIENCE},
    "pay": {"instructions": "How willing is the target audience of {subject} to pay for it?", "criteria": PAY},
    "build": {"instructions": "How feasible is it for one iOS developer to build a high-quality version of {subject} within about four weeks?",
              "criteria": BUILD},
    "kind": {"instructions": "What kind of app is `app`?", "criteria": KIND},
    "note": {"instructions": "A user rejected an app idea and wrote `note` (Turkish). Which listed reason does the note express?",
             "criteria": {**REASON_DESC, "Diğer": "None of the listed reasons"}},
}
SUBJECT_KW = "an app that serves people searching the App Store for `search_term`"
SUBJECT_APP = "`app`"

_REG = {"mtime": None, "data": None}


def registry():
    try:
        m = os.path.getmtime(REGISTRY)
        if _REG["mtime"] != m:
            _REG.update(mtime=m, data=json.load(open(REGISTRY)))
    except (FileNotFoundError, ValueError):
        _REG["data"] = None
    return _REG["data"]


def prod_specs(cap):
    reg = registry()
    if reg and cap in reg.get("caps", {}):
        c = reg["caps"][cap]
        return [c["versions"][v] for v in c["prod"]]
    return [V1[cap]]


def _fill(x, fmt):
    if isinstance(x, str):
        for k, v in fmt.items():
            x = x.replace("{" + k + "}", str(v))
        return x
    if isinstance(x, list):
        return [_fill(y, fmt) for y in x]
    if isinstance(x, dict):
        return {k: _fill(v, fmt) for k, v in x.items()}
    return x


def build_q(cap, spec, **fmt):
    return {"type": TYPES[cap], "instructions": _fill(copy.deepcopy(spec["instructions"]), fmt),
            "criteria": _fill(copy.deepcopy(spec["criteria"]), fmt)}


def add_q(qs, key, cap, specs=None, **fmt):
    """Add one question per production version (ensemble members key, key~1, key~2 ...)."""
    for n, spec in enumerate(specs or prod_specs(cap)):
        qs[key if n == 0 else f"{key}~{n}"] = build_q(cap, spec, **fmt)


def merged(answers, key):
    """Average ensemble members into one answer of the same shape."""
    members = [answers[k] for k in answers if k == key or k.startswith(key + "~")]
    if len(members) == 1:
        return members[0]
    a0 = members[0]
    if a0["type"] == "noul":
        return {"type": "noul", "noul": sum(m["noul"] for m in members) / len(members)}
    keys = a0["probabilities"].keys()
    probs = {k: sum(m["probabilities"].get(k, 0) for m in members) / len(members) for k in keys}
    if a0["type"] == "choice":
        best = max(probs, key=probs.get)
        return {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}
    score = sum(int(k) * p for k, p in probs.items())
    return {"type": "score", "score": score, "legend": a0["legend"], "probabilities": probs,
            "confidence": min(m["confidence"] for m in members)}


# ---- keyword / niche judgments -------------------------------------------------
def keyword_state(term, profession, top):
    return {"search_term": term, "profession": profession or "unknown",
            "top_results": [{"rank": i + 1, "name": a["name"], "developer": a.get("dev", ""),
                             "ratings": a["ratings"], "description": (a.get("desc") or "")[:300]}
                            for i, a in enumerate(top)]}


def judge_keyword(term, profession, top):
    q = {}
    add_q(q, "intent", "intent")
    add_q(q, "brand", "brand")
    for cap in ("audience", "pay", "build", "risk"):
        add_q(q, cap, cap, subject=SUBJECT_KW)
    for i in range(len(top)):
        add_q(q, f"rival_{i}", "rival", i=i)
    r = ask(keyword_state(term, profession, top), q)
    a = r["answers"]
    it = merged(a, "intent")
    return {"intent": it["choice"], "intentP": round(it["probabilities"][it["choice"]], 3),
            "brandP": norm(merged(a, "brand")),
            **{c: norm(merged(a, c)) for c in ("audience", "pay", "build", "risk")},
            "rivalP": [norm(merged(a, f"rival_{i}")) for i in range(len(top))], "model": r["model"]}


# ---- app judgments (features the learner weighs with your labels) --------------
APP_FEATURES = ["audience", "pay", "build", "risk"] + ["kind_" + k for k in KIND]


def app_state(a):
    return {"app": {"name": a["name"], "developer": a.get("dev", ""), "category": a.get("genre", a.get("category", "")),
                    "ratings": a.get("ratings", 0), "description": (a.get("desc") or "")[:900]}}


def judge_app(a):
    q = {}
    add_q(q, "kind", "kind")
    for cap in ("audience", "pay", "build", "risk"):
        add_q(q, cap, cap, subject=SUBJECT_APP)
    r = ask(app_state(a), q)
    an = r["answers"]
    out = {k: norm(merged(an, k)) for k in ("audience", "pay", "build", "risk")}
    kd = merged(an, "kind")
    out["kind"] = kd["choice"]
    out["kindP"] = {k: round(v, 3) for k, v in kd["probabilities"].items()}
    out["model"] = r["model"]
    return out


def app_vector(j):
    return [j["audience"], j["pay"], j["build"], j["risk"]] + [j["kindP"].get(k, 0) for k in KIND]


# ---- free-text "bad" note -> reason -------------------------------------------
def note_reason(note, reasons=None):
    q = {}
    add_q(q, "reason", "note")
    r = ask({"note": note}, q)
    a = merged(r["answers"], "reason")
    return a["choice"], a["confidence"]
