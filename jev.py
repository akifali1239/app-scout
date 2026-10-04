"""TypeSafe Jev client + the judgments App Scout asks it. API: https://docs.typesafe.ai/api.md

Jev returns typed judgments (choice / score / noul probabilities), not text. Code keeps the policy:
which answers count, thresholds and weights all live in server.py.
"""
import json, os, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


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
    """Score -> 0..1 position on its levels; noul -> probability of yes."""
    if ans["type"] == "noul":
        return round(ans["noul"], 3)
    if ans["type"] == "score":
        return round(ans["score"] / max(1, len(ans["legend"]) - 1), 3)
    return ans["choice"]


# ---- shared rubrics ----------------------------------------------------------
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


# ---- keyword / niche judgments -------------------------------------------------
INTENT = {
    "need": "A kind of tool, calculator, reference, study aid or utility; any good app doing this job would satisfy them",
    "brand": "One specific company, product, publisher or app, by its name",
    "local_business": "The app of one specific clinic, shop, school, credit union or other organization",
    "game": "A game or simulator to play for fun",
    "jobs": "Job listings, hiring or staffing",
    "other": "None of the above",
}


def judge_keyword(term, profession, top):
    state = {"search_term": term, "profession": profession or "unknown",
             "top_results": [{"rank": i + 1, "name": a["name"], "developer": a.get("dev", ""),
                              "ratings": a["ratings"], "description": (a.get("desc") or "")[:300]}
                             for i, a in enumerate(top)]}
    q = {
        "intent": {"type": "choice", "criteria": INTENT,
                   "instructions": "What is a person who types `search_term` into the App Store search box looking for?"},
        "audience": {"type": "score", "criteria": AUDIENCE,
                     "instructions": "Who mainly searches the App Store for `search_term`?"},
        "pay": {"type": "score", "criteria": PAY,
                "instructions": "How willing is the typical person searching `search_term` to pay for an app that serves it well?"},
        "build": {"type": "score", "criteria": BUILD,
                  "instructions": "How feasible is it for one iOS developer to build a high-quality app that serves `search_term` within about four weeks?"},
        "risk": {"type": "noul", "criteria": RISK,
                 "instructions": f"Would an app built to serve `search_term` likely {RISK_Q}?"},
    }
    for i in range(len(top)):
        q[f"rival_{i}"] = {"type": "noul", "instructions": f"Does `top_results[{i}]` directly serve what a person searching `search_term` wants?",
                           "criteria": {"true": "Built for this need: a real competitor",
                                        "false": "Unrelated, only loosely related, or a general app that merely mentions it"}}
    r = ask(state, q)
    a = r["answers"]
    return {"intent": a["intent"]["choice"], "intentP": round(a["intent"]["probabilities"][a["intent"]["choice"]], 3),
            "audience": norm(a["audience"]), "pay": norm(a["pay"]), "build": norm(a["build"]), "risk": norm(a["risk"]),
            "rivalP": [norm(a[f"rival_{i}"]) for i in range(len(top))], "model": r["model"]}


# ---- app judgments (features the learner weighs with your labels) --------------
KIND = {
    "utility": "A tool that does one job with the phone's sensors, camera, audio, files or calculations",
    "reference": "Reference, learning, study or exam-prep content for a topic",
    "content": "Media, entertainment, streaming, social or news",
    "game": "A game",
    "ai_wrapper": "Mainly a chat or AI-generation interface",
    "service": "Front end for a business, account or online service",
}
APP_FEATURES = ["audience", "pay", "build", "risk"] + ["kind_" + k for k in KIND]


def judge_app(a):
    state = {"app": {"name": a["name"], "developer": a["dev"], "category": a["genre"],
                     "ratings": a["ratings"], "description": a["desc"][:900]}}
    q = {
        "kind": {"type": "choice", "criteria": KIND, "instructions": "What kind of app is `app`?"},
        "audience": {"type": "score", "criteria": AUDIENCE, "instructions": "Who is `app` built for?"},
        "pay": {"type": "score", "criteria": PAY,
                "instructions": "How willing are the people `app` is built for to pay for an app like it?"},
        "build": {"type": "score", "criteria": BUILD,
                  "instructions": "How feasible is it for one iOS developer to build a competing app as good as `app` within about four weeks?"},
        "risk": {"type": "noul", "criteria": RISK, "instructions": f"Would an app like `app` likely {RISK_Q}?"},
    }
    r = ask(state, q)
    an = r["answers"]
    out = {k: norm(an[k]) for k in ("audience", "pay", "build", "risk")}
    out["kind"] = an["kind"]["choice"]
    out["kindP"] = {k: round(v, 3) for k, v in an["kind"]["probabilities"].items()}
    out["model"] = r["model"]
    return out


def app_vector(j):
    return [j["audience"], j["pay"], j["build"], j["risk"]] + [j["kindP"].get(k, 0) for k in KIND]


# ---- free-text "bad" note -> reason -------------------------------------------
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


def note_reason(note, reasons):
    crit = {r: REASON_DESC.get(r) for r in reasons}
    crit["Diğer"] = "None of the listed reasons"
    r = ask({"note": note}, {"reason": {
        "type": "choice", "criteria": crit,
        "instructions": "A user rejected an app idea and wrote `note` (Turkish). Which listed reason does the note express?"}})
    a = r["answers"]["reason"]
    return a["choice"], a["confidence"]
