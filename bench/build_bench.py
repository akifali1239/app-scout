#!/usr/bin/env python3
"""One-off: assemble bench/data.json (hand-labelled ground truth for every Jev capability).
After this, bench/data.json is the source of truth; the cloud brain appends new labelled cases to it.
Each case gets a stable split: train (few-shot examples only) / dev (optimisation) / test (promotion gate)."""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from jev_eval import INTENT_CASES, RIVAL_CASES, RISK_CASES, PAIRS, NOTE_CASES  # noqa: E402
from jev_eval2 import LABELS, EXTRA_BRANDS, TR  # noqa: E402


def split(key):
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 25 else "dev" if h < 62 else "test"


def top_of(term, kw):
    k = kw.get(term)
    return [{"name": a["name"], "dev": a.get("dev", ""), "ratings": a["ratings"], "desc": (a.get("desc") or "")[:200]}
            for a in (k["top"] if k else [])][:10]


kw = json.load(open(os.path.join(os.path.dirname(HERE), "keywords.json")))
D = {}

# intent: term -> one of need/brand/local_business/game/jobs/other
D["intent"] = [{"term": t, "label": lab} for lab, ts in INTENT_CASES.items() for t in ts] + [
    {"term": t, "label": l} for t, l in [
        ("acupuncture", "need"), ("pocket prep", "brand"), ("pilot flying j", "brand"), ("fireman plumber", "game"),
        ("plumber jobs near me", "jobs"), ("speech therapy delhi", "local_business"),
        ("elite physical therapy clinic", "local_business"), ("trucker path", "brand"), ("truck simulator", "game"),
        ("nurse jobs", "jobs"), ("kaynak sembolleri", "need"), ("trendyol", "brand"), ("acıbadem hastanesi", "local_business"),
        ("machinist speeds and feeds", "need"), ("massage therapy exams", "need"), ("somatic massage therapy pc", "local_business")]]

# brand: real mined terms with their real top-10 as evidence; label True = proper/brand name
BRAND_MORE = {
    "personal trainer: home workout": True, "express chiropractic llc": True, "plumber license prep: exam": True,
    "somatic massage therapy pc": True, "massage therapy exams": False, "pilot driver app": True,
    "paramedic trauma review": False, "paramedic pocket prep": True, "pharmacy technician educator": False,
    "nutricrm - dietitian software": True, "white noise": False, "e6b flight computer": False,
    "pharmacy technician games": False, "fireman plumber : rescue now !": True, "color match identifier": False,
    "muv chiropractic & yoga pc": True, "acupuncture atlas 3d": False, "pilot express": True,
    "medical terminology": False, "journeyman plumber exam prep": False, "physical therapy by phoenix": True,
    "elite physical therapy clinic": True, "physical therapy tracker": False,
    "midwifery study guide- 1400 notes, quiz & concepts": True, "dental hygiene exam prep": False,
    "massage therapy mblex prep": False, "aida: ai dietitian + groceries": True, "dietitian niveta's diet clinic": True,
    "cnm nurse midwife exam prep": False, "acupuncture points, cool index": True, "speech therapy - word tube": True,
    "rda dental assistant exam prep": False, "personal trainer prep: test": True, "machinist speeds & feeds": False,
    "countdown widget": False, "speed test internet wifi scan": False}
D["brand"] = ([{"term": t, "top": top_of(t, kw), "label": lab == "brand"} for t, lab in LABELS.items()]
              + [{"term": t, "top": top_of(t, kw), "label": True} for t in EXTRA_BRANDS]
              + [{"term": t, "top": top_of(t, kw), "label": v} for t, v in BRAND_MORE.items()])

# rival: groups (term + candidate apps); label per app = 1 if built for this need
D["rival"] = [{"term": t, "apps": [{"name": a["name"], "dev": a["dev"], "ratings": a["ratings"], "desc": a["desc"]} for a, _, _ in items],
               "labels": [y for _, y, _ in items], "kinds": [k for _, _, k in items]} for t, items in {**RIVAL_CASES, **TR}.items()]
D["rival"].append({"term": "welding symbols", "apps": [
    {"name": "Welder's Handbook", "dev": "Dev", "ratings": 100, "desc": "All-in-one welding app: calculators, electrode charts, and a complete welding symbols chapter with quizzes."},
    {"name": "CWI Exam Prep", "dev": "Dev", "ratings": 100, "desc": "Practice exams for the Certified Welding Inspector test, including questions on welding symbols."},
    {"name": "Blueprint Reading for Trades", "dev": "Dev", "ratings": 100, "desc": "Learn to read construction and fabrication drawings, including welding symbols."}],
    "labels": [1, 0, 0], "kinds": ["suite", "exam-covers", "broader"], "id": "welding symbols (partial)"})

# risk: app description -> restricted by App Review because of what the app itself does
D["risk"] = [{"app": a, "label": bool(y)} for a, y in RISK_CASES] + [{"app": a, "label": y} for a, y in [
    ("Symptom checker: enter your symptoms and get a list of the most likely diagnoses.", True),
    ("Medication reminder that alerts you when to take the pills your doctor prescribed.", False),
    ("Peptide dosage calculator: tells you how many units of semaglutide to inject.", True),
    ("Calorie counter and macro tracker.", False),
    ("Hearing test that diagnoses hearing loss so you don't need an audiologist.", True),
    ("Copy trading: automatically mirror the trades of top investors with your own money.", True),
    ("Legal dictionary of law terms for paralegal students.", False),
    ("Create a legally binding will valid in all 50 states, with AI lawyer advice.", True),
    ("Daily Bible verse widget.", False),
    ("Stock market news and charts.", False)]]

# score dims: ordered pairs, first should score higher
D["pairs"] = [{"dim": dim, "hi": a, "lo": b} for dim, ps in PAIRS.items() for a, b in ps]

# note -> reason (Turkish free text)
D["note"] = [{"note": n, "label": r} for n, r in NOTE_CASES] + [{"note": n, "label": r} for n, r in [
    ("rakipler zayıf ama bence kimse para ödemez", "Para yapmaz"),
    ("bunu yapmak için tıbbi lisans gerekir, apple onaylamaz", "Ret riski"),
    ("listede bu üçüncü kez çıkıyor", "Çok benzer"),
    ("güzel fikir ama bu nişte milyon indirmeli 5 app var", "Çok rekabet / doygun"),
    ("benim tarzım değil, bu bir oyun", "Alakasız"),
    ("3D model ve büyük içerik ekibi lazım", "Yapması zor"),
    ("ikonu ve ekran görüntüleri 2015'ten kalma", "Kalitesiz / eski"),
    ("çok sıradan, heyecan yok", "Sıkıcı fikir")]]

# kind: app -> utility/reference/content/game/ai_wrapper/service
D["kind"] = [{"app": {"name": n, "category": g, "desc": d}, "label": l} for n, g, d, l in [
    ("Pill Reminder & Medicine Alarm", "Health & Fitness", "Reminders for medications and supplements with refill alerts.", "utility"),
    ("Increase Volume: Sound Enhance", "Music", "Turn up the volume and hear every detail with an audio booster.", "utility"),
    ("PREMIER Credit Card", "Finance", "Manage your PREMIER Bankcard credit card account, pay bills and see transactions.", "service"),
    ("Color Pick Camera: HEX RGB", "Graphics & Design", "See the HEX code of the color you aim the camera at.", "utility"),
    ("Mushroom Identifier: Plant ID", "Utilities", "Point your camera at a mushroom to identify the species.", "utility"),
    ("TurboScan: document scanner", "Business", "Turns your iPhone into a scanner for documents and receipts.", "utility"),
    ("Music AI: Playlist Maker", "Music", "Describe a mood and AI generates a Spotify playlist for you.", "ai_wrapper"),
    ("Woto - Relationship Companion", "Utilities", "AI texting assistant that writes replies for your conversations.", "ai_wrapper"),
    ("AI Helper - AI Chatbot", "Productivity", "Chat with AI to write essays, emails and speeches.", "ai_wrapper"),
    ("Qibla Compass", "Navigation", "Compass that points to the Qibla direction.", "utility"),
    ("Pocket Spy: Anti-Theft Alarm", "Utilities", "Sounds an alarm when someone moves or unplugs your phone.", "utility"),
    ("Classic Bubble Pop-Ball Games", "Games", "Beat challenges, win levels and reach high scores.", "game"),
    ("Eenies at War", "Games", "Massive multiplayer online artillery warfare.", "game"),
    ("Zero: Fasting & Food Tracker", "Health & Fitness", "Fasting timer and nutrition tracker.", "utility"),
    ("Davis's Drug Guide", "Medical", "Monographs for 5,000+ drugs: dosing, interactions and nursing considerations.", "reference"),
    ("NCLEX RN Mastery", "Education", "3,000 practice questions with rationales for the nursing licensing exam.", "reference"),
    ("Acupuncture Atlas 3D", "Medical", "Locate acupuncture points on a 3D body with indications.", "reference"),
    ("Manga Reader Plus", "Books", "Read thousands of manga series online.", "content"),
    ("Radio FM", "Music", "Listen to thousands of live radio stations.", "content"),
    ("Daily Bible Verse", "Lifestyle", "A verse and a prayer every day in a calm feed.", "content"),
    ("Hinge Health", "Health & Fitness", "Virtual physical therapy program from licensed PTs, covered by your employer.", "service"),
    ("Find my Friends, Family Phone", "Social Networking", "See where your family is on a map and chat with them.", "service"),
    ("Uber Eats", "Food & Drink", "Order food delivery from local restaurants.", "service"),
    ("Welding Symbols Guide", "Reference", "Reference of welding symbols with pictures and a quiz.", "reference")]]

# proportional, stratified split: within each capability (and label), cases are ordered by a stable hash and dealt
# round-robin train/dev/test/dev/test... so every split gets every label and dev/test are never empty.
DEAL = ["dev", "test", "train", "dev", "test", "test", "dev", "train"]
for cap, rows in D.items():
    def key(r):
        return hashlib.md5((cap + json.dumps({k: v for k, v in r.items() if k not in ("label", "labels", "kinds", "top", "split")},
                                             sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
    groups = {}
    for r in rows:
        groups.setdefault(json.dumps(r.get("label", r.get("dim", ""))), []).append(r)
    for g in groups.values():
        for n, r in enumerate(sorted(g, key=key)):
            r["split"] = DEAL[n % len(DEAL)]
json.dump(D, open(os.path.join(HERE, "data.json"), "w"), ensure_ascii=False, indent=1)
print({cap: {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")} for cap, rows in D.items()})
