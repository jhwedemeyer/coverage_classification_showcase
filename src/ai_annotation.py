"""
Annotate the validation sample with a large language model (Google Gemini).

Fills a copy of the annotation sheet written by generate_test_sample.py, leaving the original
for human annotation. For each of the five impact types the model returns a graded label:
"about" (the article reports on the event), "mention" (only in passing) or "none". This lets
precision be reported under a strict and a lenient definition from a single API run. The model
also names the place and country of the event, to validate the location attribution.

Requests go through the Gemini Batch API (asynchronous, half price), so the work is split into
resumable steps:

    python src/ai_annotation.py submit [--smoke 20]   # send articles not yet annotated
    python src/ai_annotation.py status                # check the batch jobs
    python src/ai_annotation.py collect               # download finished results
    python src/ai_annotation.py merge                 # write the annotated CSV

Add --sample continent to annotate the continent sample instead. After changing CODEBOOK, move
ai_annotation_raw.jsonl aside before submitting again, or nothing will be re-annotated.

Requires google-genai and GEMINI_API_KEY (except for merge). Run from the repository root.
"""

import argparse
import functools
import json
import os
import sys
from datetime import datetime, timezone

import countrynames
import pandas as pd

# Batch API prices per 1M tokens (input/output) and the resulting cost for the full sample:
# gemini-3.1-flash-lite  $0.125/$0.75  ~$0.90
# gemini-2.5-flash-lite  $0.05 /$0.20  ~$0.30
# gemini-3.6-flash       $0.75 /$3.75  ~$4.50
MODEL = "gemini-3.1-flash-lite"

TESTSAMPLE = "data/testsample/"
ANNOTATION = TESTSAMPLE + "test_sample_annotation.csv"
KEY = TESTSAMPLE + "test_sample_key.csv"
OUTFILE = TESTSAMPLE + "test_sample_annotation_ai.csv"
REQUESTS = TESTSAMPLE + "ai_annotation_requests.jsonl"
RAW = TESTSAMPLE + "ai_annotation_raw.jsonl"
JOBS = TESTSAMPLE + "ai_annotation_batch.json"

# The samples generate_test_sample.py draws, each with its own sheet, key and batch bookkeeping,
# so that what is annotated and in flight is tracked per sample. The paths above are those of
# the main sample, which the evaluation scripts import; --sample switches them all at once.
SAMPLES = {
    "main": {"ANNOTATION": ANNOTATION, "KEY": KEY, "OUTFILE": OUTFILE,
             "REQUESTS": REQUESTS, "RAW": RAW, "JOBS": JOBS},
    "continent": {
        "ANNOTATION": TESTSAMPLE + "test_sample_continent_annotation.csv",
        "KEY": TESTSAMPLE + "test_sample_continent_key.csv",
        "OUTFILE": TESTSAMPLE + "test_sample_continent_annotation_ai.csv",
        "REQUESTS": TESTSAMPLE + "ai_annotation_continent_requests.jsonl",
        "RAW": TESTSAMPLE + "ai_annotation_continent_raw.jsonl",
        "JOBS": TESTSAMPLE + "ai_annotation_continent_batch.json",
    },
}


def use_sample(name):
    """Point every path of the module at the files of sample *name*."""
    globals().update(SAMPLES[name])
    globals()["SAMPLE"] = name


SAMPLE = "main"


def sample_flag():
    """The --sample argument that addresses the current sample, for the hints printed."""
    return "" if SAMPLE == "main" else " --sample " + SAMPLE

# Hazard classes of the panel, identical to generate_test_sample.py
EVENTS = ["Flood", "Wildfire", "Heat", "Storm", "Cold"]

GRADES = ["about", "mention", "none"]

# "strict": only "about" counts as the article reporting on the hazard
# "lenient": "about" and "mention" both count, which is closest to what the dictionary does
STRICTNESS = "lenient"

MAX_CHARS = 20000          # articles are truncated to this length (39 of 3,300 rows)


CODEBOOK = """\
You annotate German newspaper articles published between 1990 and 2022 for a study. Your labels \
validate a keyword classifier, so judge only what the article actually says -- never guess from \
the headline alone, and never assume an article must be about a hazard. Most of these articles are \
about something else entirely, and "none" on all five types is the ordinary answer.

Grade every hazard type in two steps, always in this order:

1. Does the article refer to a concrete, real, extreme event of that type -- one that actually \
happened, somewhere, at some time? If not, the grade is "none". The exclusions listed further down \
decide this step and they override everything else: a passage that falls under one of them is \
"none" however vividly it is written and however often the hazard word appears.
2. Only once step 1 says yes, decide how much of the article that event takes up: "about" or \
"mention".

Never arrive at "mention" merely because a hazard word appears briefly. Brevity decides between \
"about" and "mention"; it never turns a non-event into an event.

Only *extreme* events count. Ordinary weather, a season that ran warmer, colder, \
wetter or drier than usual, and everyday inconvenience are not hazard events, however often the \
article mentions them.

Grade each of these five hazard types independently:

- flood: floods, flash floods, damaging heavy rainfall and its consequences, storm surges, \
rivers and lakes above flood level. Not the ordinary tide, a river merely carrying water, or rain \
that fell without damage or inundation.
- wildfire: forest, bush, grass, heath and moor fires burning out of control over open land. What \
burns has to be vegetation. A building, factory, warehouse, hall, shed, vehicle, ship, landfill, \
waste-depot, refugee-camp or city fire is not a wildfire, however large or deadly it was, and \
neither is a firestorm caused by bombing, nor a single tree or structure set alight by lightning. \
Where the article does not make clear that vegetation burned across open land, grade "none".
- heat: heatwaves, extreme heat, record high temperatures and their consequences. Not temperatures \
that are ordinary for that region and season: a Sahel summer, a desert, the tropics. Heat that is \
simply the local climate is "none", however extreme it sounds to a German reader.
- storm: severe storms, hurricanes, tropical cyclones, typhoons, tornadoes, winter storms and \
damaging wind events. Not wind that merely blew: gusts that made a match, a regatta or a crossing \
difficult are not a storm event. A winter storm belongs to storm and to cold at once: grade it on \
both, never choose one and put the other at "none".
- cold: cold waves and cold snaps, extreme cold, record low temperatures, blizzards, winter \
storms, heavy snowfall and severe icing. A winter storm counts here whether the article dwells on \
the cold and the snow or on the wind and the damage: a named one -- "Wintersturm Lothar", \
"Schneesturm Jonas", "Orkan Kyrill" -- is a cold event on its name alone. Not winter being wintry: \
snow that simply fell, a frost night, or the ordinary climate of a polar region or a research \
station.

Several hazards belong to none of the five types. An article reporting only these scores "none" on \
all five: drought and water scarcity, hail, sand and dust storms, avalanches, landslides and \
mudslides, earthquakes, volcanic eruptions, and tsunamis.

Use one of three grades for each type:

- "about": the article substantively reports on one or more concrete, real, extreme events of \
that type -- its occurrence, its consequences, the response to it, or its aftermath. The event \
is a subject of the article, not an aside. Reserve "about" for what a reader would call an article \
about that event: several paragraphs, or a short piece given over to it entirely.
- "mention": a concrete, real, extreme event of that type is referred to, but only in passing -- \
a single clause or sentence, an example inside an article about something else, a historical \
comparison, a list entry. An event named in one clause of a long article on another subject is \
"mention", not "about", even where the event itself was a major disaster.
- "none": no concrete, real, extreme event of that type is referred to at all.

"mention" is for an extreme event reported briefly, never for weather that falls short of \
extreme. Ordinary weather in passing is "none".

Grade "none", not "mention", in all of these cases. They override the grades above:

- Metaphorical or idiomatic use of hazard vocabulary: "eine Flut von Anfragen", "Dürre" for a \
team's goal drought, "Sturm der Entrüstung", "eiskalt" for a person, "Hitzeschlacht" for a match, \
"wie ein Waldbrand" for anything that spreads, a team named Hurricanes.
- Abstract discussion of climate change, climate policy, climate science, emissions or energy \
that does not report a specific hazard event, even when it names hazard types in general terms.
- Risk, prevention, insurance, infrastructure or preparedness pieces that discuss hazards \
generically without a specific event: danger levels and warning stages with no fire or flood \
behind them ("Waldbrandgefahrenstufe fünf", "erhöhte Hochwassergefahr"), hazard maps, seasonal \
outlooks, and the planning, funding, public hearing about or building of dykes, barriers, \
retention basins and firebreaks.
- Weather forecasts and ordinary weather reports that do not describe an extreme event, including \
forecast heat, snow or wind still to come that has not yet done anything.
- Unremarkable weather and seasonal variation, even where the article calls it unusual: a cold \
or wet spring, a mild winter, a rainy summer, a warm day, a chilly night, snow that simply fell, \
rain that simply fell, wind that simply blew, a season's temperature or rainfall averages. "Es \
war ein kalter Fruehling" is not a cold wave; "der verregnete Sommer" is not a flood; "es \
regnete stark" without damage or flooding is not a flood.
- Weather as the backdrop to something else -- the conditions at a match, a regatta, a festival, a \
concert, a demonstration, a holiday, a harvest, a construction site, a dinner, a journey. This \
holds even where the weather system is named and was severe elsewhere: where the article's own \
words say only that it was stormy, snowy or hot while something else went on, grade "none". Grade \
an event only where the article itself reports that event.
- A climate fact rather than an event: a standing all-time figure quoted for reference -- the \
lowest temperature ever measured somewhere, how cold it gets at a research station, what a \
region's summers are like, a record on a museum label or in an almanac. A record actually reached \
in the episode the article reports is an event, not a climate fact.
- Consequences with no event behind them in the article: crop yields, energy demand, heating \
costs, tourism figures or clothing sales attributed to the weather in general.
- Fiction, film and book plots, satire, hypothetical scenarios, myth and legend. A blizzard in the \
plot of a film and the flood of the Noah story are "none".

None of the exclusions above apply to an event the article names or dates as an event in its own \
right: a "Hitzewelle", the "Jahrhundertsommer 2003", an "Orkan Kyrill", a "Flutkatastrophe", a \
heading announcing heatwaves and record temperatures, an episode whose damage, deaths, evacuations \
or emergency response the article reports. Such an event is concrete however briefly it is named \
and whatever else the article is mostly about, and the exclusion for abstract climate talk never \
reaches it -- that exclusion is for hazard types named in general terms.

This guard does not loosen the exclusions themselves. A hazard word applied to the conditions \
around something else -- a storm around a house, wind at a championship, snow during a journey -- \
names no event and stays "none".

Before you grade anything other than "none", read the note you are about to write. If it says that \
the article mentions weather, conditions, risks or hazards in general rather than reporting a \
concrete event, then the grade you have just reasoned your way to is "none".

When in doubt about severity, ask whether a reader would call the episode a disaster or an \
extreme weather event, or merely bad weather. Grade "none" if it is merely bad weather. When in \
doubt between "about" and "mention", choose "mention".

Events anywhere in the world count, not only in Germany. Events from any year count, including \
historical ones. An article may be graded for several hazard types at once; grade each on its own \
merits.

For the hazard events you graded "about" or "mention", say where they happened.

"location" is the single place the article's hazard reporting centres on -- the one place you \
would name if asked where the event in this article happened. Report the place the hazard \
occurred, not every place the article names. The city a politician spoke in, the dateline, the \
city hosting a conference or a summit, the seat of an aid agency or a relief fund, the home \
country of someone quoted, the country sending help or firefighters, and the place of a \
comparison event are not the location of this event unless the event also happened there.

- "place": the place as the article names it, e.g. "Elbe", "Kalifornien", "New Orleans". Use an \
empty string when the article reports a hazard but never says where it happened.
- "country": the country that place lies in, as its common English name, e.g. "Germany", \
"United States". Use an empty string when the place lies in no single country -- an ocean, a \
whole continent, "weltweit" -- or when there is no place.
- "hazards": which of the five hazard types occurred at that place.

Put any further places a hazard occurred in "other_locations", most prominent first, with the same \
fields. Leave it empty when there are none.

Leave "location" and "other_locations" empty when every hazard is graded "none".

Set "location_unclear" to true when the article reports a hazard but you cannot tell where it \
happened -- several candidate places with nothing to choose between them, or a name too ambiguous \
to place. An article that runs through several events in several countries with none of them at \
its centre is unclear: set the flag rather than picking one of them arbitrarily. An article that \
simply never states a location is not unclear: give it an empty place.

Set "unclear" to true only when the text is genuinely undecidable -- it is truncated mid-sentence \
before the subject is clear, it is garbled, or it is not an article at all. Do not use "unclear" \
merely because the case is a borderline one between two grades; pick the closer grade instead.

Write "notes" as one short English sentence justifying the grades, naming the event if there is \
one. Where a grade turns on one of the exclusions above, say which one.
"""


"""Response schema

The Gemini responseSchema is a subset of OpenAPI 3.0, not JSON Schema: type names are the
uppercase proto enum names and additionalProperties is not among the supported fields.
propertyOrdering fixes the key order, which the API documents as affecting output quality.
"""


def response_schema():
    hazards = [event.lower() for event in EVENTS]
    properties = {hazard: {"type": "STRING", "enum": GRADES} for hazard in hazards}
    location = {
        "type": "OBJECT",
        "properties": {
            "place": {"type": "STRING"},
            "country": {"type": "STRING"},
            "hazards": {"type": "ARRAY", "items": {"type": "STRING", "enum": hazards}},
        },
        "required": ["place", "country", "hazards"],
        "propertyOrdering": ["place", "country", "hazards"],
    }
    properties["location"] = location
    properties["other_locations"] = {"type": "ARRAY", "items": location}
    properties["location_unclear"] = {"type": "BOOLEAN"}
    properties["unclear"] = {"type": "BOOLEAN"}
    properties["notes"] = {"type": "STRING"}
    ordering = hazards + [
        "location", "other_locations", "location_unclear", "unclear", "notes"]
    return {
        "type": "OBJECT",
        "properties": properties,
        "required": ordering,
        "propertyOrdering": ordering,
    }


"""Countries"""


@functools.lru_cache(maxsize=None)
def to_iso3(country):
    """The ISO3 code of a country name, None where it names no single country.

    countrynames.to_code_3 is the call gen_coverage_days.py resolves the geocoder's country
    with, so the codes here and the ones in its GID_0 column are in the same namespace and can
    be compared directly. Oceans, continents and empty strings resolve to None here exactly as
    they are skipped there, so an event the model placed in "the Atlantic" carries no country
    rather than a wrong one.
    """
    if not country:
        return None
    try:
        return countrynames.to_code_3(country)
    except Exception:
        return None


def format_locations(locations):
    """Pipe-joined "place (code)" rendering of the secondary locations, for eyeballing.

    Not-yet-annotated rows carry NaN rather than a list after merge's left join, and NaN is
    truthy, so the type is checked rather than the value.
    """
    if not isinstance(locations, list):
        return ""
    return "|".join(
        "{} ({})".format(entry["place"], entry["iso3"] or entry["country"] or "?")
        for entry in locations
        if entry.get("place") or entry.get("country")
    )


"""Requests"""


def article_text(article):
    """Render one article the way the annotator sees it, without the classifier outcome."""
    text = str(article.text) if pd.notna(article.text) else ""
    truncated = text[:MAX_CHARS]
    if len(text) > MAX_CHARS:
        truncated += "\n[...article truncated...]"
    return (
        "Date: {}\nNewspaper: {}\nTitle: {}\n\n{}".format(
            article.date, article.journal, article.title, truncated
        )
    )


def build_request(article, schema):
    return {
        "key": article.sample_id,
        "request": {
            "system_instruction": {"parts": [{"text": CODEBOOK}]},
            "contents": [
                {"role": "user", "parts": [{"text": article_text(article)}]}
            ],
            "generation_config": {
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        },
    }


def write_requests(sample, path=None):
    path = path or REQUESTS
    schema = response_schema()
    with open(path, "w") as stream:
        for article in sample.itertuples():
            stream.write(json.dumps(build_request(article, schema), ensure_ascii=False) + "\n")
    return path


"""Raw annotations already on disk"""


def load_raw(path=None):
    path = path or RAW
    if not os.path.exists(path):
        return {}
    annotations = {}
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if line:
                record = json.loads(line)
                annotations[record["sample_id"]] = record
    return annotations


def append_raw(records, path=None):
    path = path or RAW
    with open(path, "a") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


"""Batch job bookkeeping, so a run can be resumed from a new shell"""


def load_jobs(path=None):
    path = path or JOBS
    if not os.path.exists(path):
        return {"jobs": []}
    with open(path) as stream:
        return json.load(stream)


def save_jobs(jobs, path=None):
    path = path or JOBS
    with open(path, "w") as stream:
        json.dump(jobs, stream, indent=2)


def submitted_keys(path=None):
    """The sample_ids sitting in a submitted but not yet collected batch job.

    The daily batch quota only takes a fraction of the sample at a time, so the run is split
    across several jobs. Between submitting a chunk and collecting it the articles are in
    neither the raw file nor a finished job, and without this they would look pending and be
    sent -- and paid and queued for -- a second time. A job that failed or expired can be
    released by setting its "collected" to true in the jobs file; submit then offers its
    articles again.
    """
    keys = set()
    for job in load_jobs(path)["jobs"]:
        if not job.get("collected"):
            keys.update(job.get("keys") or [])
    return keys


def client():
    try:
        from google import genai
    except ImportError:
        sys.exit("google-genai is not installed. Run: pip install google-genai")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("GEMINI_API_KEY is not set. Get one at https://aistudio.google.com/apikey")
    return genai.Client()


"""Subcommands"""


def cmd_submit(args):
    from google.genai import types

    sample = pd.read_csv(ANNOTATION)
    annotated = set(load_raw())
    in_flight = submitted_keys()
    pending = sample[~sample.sample_id.isin(annotated | in_flight)]
    if args.smoke:
        pending = pending.head(args.smoke)
    if not len(pending):
        print("Nothing to submit: all {:,} articles are already annotated.".format(len(sample)))
        return

    print("Submitting {:,} of {:,} articles ({:,} already annotated, {:,} in flight)".format(
        len(pending), len(sample), len(annotated), len(in_flight)))
    write_requests(pending)

    genai_client = client()
    uploaded = genai_client.files.upload(
        file=REQUESTS,
        config=types.UploadFileConfig(
            display_name="ai_annotation_requests", mime_type="jsonl"),
    )
    label = "smoke-{}".format(args.smoke) if args.smoke else "full"
    try:
        batch_job = genai_client.batches.create(
            model=MODEL,
            src=uploaded.name,
            config={"display_name": "ai_annotation-" + label},
        )
    except Exception as error:
        # The Batch API is not available on the free tier
        if "FAILED_PRECONDITION" in str(error):
            sys.exit(
                "The Batch API rejected the job with FAILED_PRECONDITION. This key's project "
                "has no billing enabled -- batching starts at Tier 1, the free tier serves "
                "interactive requests only. Enable billing on the project at "
                "https://aistudio.google.com/apikey and submit again; the sample costs about "
                "$0.90 on {}.\n\n{}".format(MODEL, error))
        raise

    jobs = load_jobs()
    jobs["jobs"].append({
        "name": batch_job.name,
        "label": label,
        "model": MODEL,
        "n_requests": int(len(pending)),
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collected": False,
        "keys": list(pending.sample_id),
    })
    save_jobs(jobs)
    print("Submitted {} ({} requests) as {}".format(label, len(pending), batch_job.name))
    print("Check with: python src/ai_annotation.py status" + sample_flag())


def cmd_status(args):
    jobs = load_jobs()
    if not jobs["jobs"]:
        sys.exit("No batch job has been submitted yet.")
    genai_client = client()
    for job in jobs["jobs"]:
        batch_job = genai_client.batches.get(name=job["name"])
        print("{}  {}  {} requests  collected={}".format(
            job["name"], batch_job.state.name, job["n_requests"], job["collected"]))


def parse_response(key, payload):
    """Pull the model's JSON out of one batch output line and validate it against the schema."""
    response = payload.get("response")
    if response is None:
        return None, payload.get("error") or payload.get("status") or "no response"

    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None, "unexpected response shape"

    try:
        annotation = json.loads(text)
    except json.JSONDecodeError:
        return None, "response is not valid JSON"

    record = {"sample_id": key, "model": MODEL}
    for event in EVENTS:
        grade = annotation.get(event.lower())
        if grade not in GRADES:
            return None, "invalid grade for {}: {!r}".format(event.lower(), grade)
        record[event.lower()] = grade
    hazard_keys = [event.lower() for event in EVENTS]

    def clean_location(entry):
        """Normalise one location object, resolving its country to the pipeline's ISO3."""
        if not isinstance(entry, dict):
            return None
        country = str(entry.get("country") or "").strip()
        return {
            "place": str(entry.get("place") or "").strip(),
            "country": country,
            "iso3": to_iso3(country),
            # a hazard name the model invented is dropped rather than failing the article, so
            # one stray value does not discard an otherwise usable annotation
            "hazards": [h for h in (entry.get("hazards") or []) if h in hazard_keys],
        }

    location = clean_location(annotation.get("location") or {})
    if location is None:
        return None, "location is not an object"
    record["location"] = location

    others = annotation.get("other_locations") or []
    if not isinstance(others, list):
        return None, "other_locations is not a list"
    record["other_locations"] = []
    for entry in others:
        cleaned = clean_location(entry)
        if cleaned is None:
            return None, "other_locations entry is not an object"
        record["other_locations"].append(cleaned)

    record["location_unclear"] = bool(annotation.get("location_unclear", False))
    record["unclear"] = bool(annotation.get("unclear", False))
    record["notes"] = str(annotation.get("notes", ""))

    usage = response.get("usageMetadata") or {}
    record["input_tokens"] = usage.get("promptTokenCount", 0)
    record["output_tokens"] = usage.get("candidatesTokenCount", 0)
    return record, None


def cmd_collect(args):
    jobs = load_jobs()
    outstanding = [job for job in jobs["jobs"] if not job["collected"]]
    if not outstanding:
        sys.exit("No uncollected batch job in {}. Run submit{} first, or merge to rebuild "
                 "the CSV.".format(JOBS, sample_flag()))

    genai_client = client()
    annotated = set(load_raw())

    for job in outstanding:
        batch_job = genai_client.batches.get(name=job["name"])
        state = batch_job.state.name
        if state != "JOB_STATE_SUCCEEDED":
            print("{} is {}, skipping".format(job["name"], state))
            continue

        content = genai_client.files.download(file=batch_job.dest.file_name)
        records, failures = [], []
        for line in content.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key = payload.get("key")
            if key in annotated:
                continue
            record, error = parse_response(key, payload)
            if record is None:
                failures.append((key, error))
            else:
                records.append(record)
                annotated.add(key)

        append_raw(records)
        job["collected"] = True
        save_jobs(jobs)

        input_tokens = sum(r["input_tokens"] for r in records)
        output_tokens = sum(r["output_tokens"] for r in records)
        print("{}: collected {:,} annotations, {:,} failed".format(
            job["name"], len(records), len(failures)))
        print("  tokens: {:,} in, {:,} out".format(input_tokens, output_tokens))
        for key, error in failures[:10]:
            print("  FAILED {}: {}".format(key, error))
        if len(failures) > 10:
            print("  ... and {:,} more".format(len(failures) - 10))
        if failures:
            print("  Re-run submit{} to retry the failed articles.".format(sample_flag()))

    total = len(load_raw())
    sample_size = len(pd.read_csv(ANNOTATION, usecols=["sample_id"]))
    print("{:,} of {:,} articles annotated".format(total, sample_size))


def cmd_merge(args):
    annotations = load_raw()
    if not annotations:
        sys.exit("No annotations on disk yet. Run submit and collect first.")

    sample = pd.read_csv(ANNOTATION)
    grades = pd.DataFrame.from_records(list(annotations.values()))

    merged = sample.merge(
        grades[["sample_id"] + [e.lower() for e in EVENTS]
               + ["location", "other_locations", "location_unclear", "unclear", "notes"]],
        on="sample_id", how="left", suffixes=("", "_ai"))

    """Derive the binary annotation columns from the grades"""
    counts_as_hazard = ["about"] if STRICTNESS == "strict" else ["about", "mention"]
    labelled = merged[EVENTS[0].lower()].notna()

    for event in EVENTS:
        grade = merged[event.lower()]
        merged["label_" + event.lower()] = (
            grade.isin(counts_as_hazard).astype("Int64").where(labelled))

    hazard = merged[["label_" + e.lower() for e in EVENTS]].sum(axis=1)
    merged["label_no_hazard"] = (hazard == 0).astype("Int64").where(labelled)
    merged["label_unclear"] = merged["unclear"].astype("boolean").astype("Int64")
    merged["notes"] = merged["notes_ai"] if "notes_ai" in merged else merged["notes"]

    """Fill the location columns.

    The sheet holds one location per article, matching gen_coverage_days.py, which attributes
    an article to the single country of the location closest to an impact keyword. label_country
    is the ISO3 code so that it compares directly against the key's location_GID_0; the model's
    own wording of the country is kept beside it in country_name.

    label_no_location marks an article that reports a hazard but names no place for it, which is
    what the pipeline records as location_status "no_location". It stays 0 for an article that
    reports no hazard at all -- there is no location to be missing.
    """
    place = merged["location"].map(
        lambda loc: loc["place"] if isinstance(loc, dict) else "")
    country_name = merged["location"].map(
        lambda loc: loc["country"] if isinstance(loc, dict) else "")
    iso3 = merged["location"].map(
        lambda loc: loc["iso3"] if isinstance(loc, dict) else None)

    merged["label_location"] = place.where(labelled)
    merged["label_country"] = iso3.where(labelled)
    merged["label_no_location"] = (
        ((hazard > 0) & (place == "")).astype("Int64").where(labelled))
    merged["label_location_unclear"] = (
        merged["location_unclear"].astype("boolean").astype("Int64"))
    merged["country_name"] = country_name.where(labelled)
    merged["other_locations"] = merged["other_locations"].map(format_locations).where(labelled)

    """Keep the column order of the annotation sheet, append the grades and the audit columns"""
    merged = merged.rename(columns={e.lower(): "grade_" + e.lower() for e in EVENTS})
    columns = list(sample.columns) + ["grade_" + e.lower() for e in EVENTS]
    columns += ["country_name", "other_locations"]
    merged[columns].to_csv(OUTFILE, index=False)

    unfilled = [c for c in sample.columns
                if c.startswith("label_") and merged.loc[labelled, c].isna().all()]
    missing = int((~labelled).sum())
    print("Wrote {:,} rows to {} (STRICTNESS={}, {:,} not yet annotated)".format(
        len(merged), OUTFILE, STRICTNESS, missing))
    if unfilled:
        print("WARNING: no value produced for {}".format(", ".join(unfilled)))

    report(merged[labelled])


def report(merged):
    """Sanity checks on the annotations, against the unblinding key where it helps.

    The share of articles the model marks as reporting on a hazard should be high within each
    classified stratum and low within the unclassified one. A classified stratum coming back
    mostly no_hazard points at the prompt, not at the classifier.

    The location half is checked against the key, which carries what the NER-and-geocoding
    pipeline attributed each article to. Agreement on location_GID_0 is the headline number;
    the disagreement rows are where the annotation earns its keep, so their counts are printed
    separately rather than folded into one accuracy figure.
    """
    hazard = merged[["label_" + e.lower() for e in EVENTS]].sum(axis=1) > 0

    if not os.path.exists(KEY):
        return
    key = pd.read_csv(KEY)
    joined = merged.merge(key[["sample_id", "stratum", "location_GID_0", "location_closest",
                               "location_status"]], on="sample_id", how="inner")
    if not len(joined):
        return

    print("\nShare annotated as reporting on any hazard, by stratum:")
    for stratum, group in joined.groupby("stratum"):
        share = 1 - group["label_no_hazard"].mean()
        print("  {:<24} {:>6.1%}  (n = {:,})".format(stratum, share, len(group)))
    print("  {:<24} {:>6.1%}".format("unclear", joined["label_unclear"].mean()))

    print("\nLocation, annotated vs. pipeline:")
    print("  hazard articles with a country   {:>6.1%}  (n = {:,})".format(
        (merged.loc[hazard, "label_country"].notna()).mean(), int(hazard.sum())))
    print("  hazard articles, no place named  {:>6.1%}".format(
        merged.loc[hazard, "label_no_location"].fillna(0).mean()))
    print("  location unclear                 {:>6.1%}".format(
        merged["label_location_unclear"].fillna(0).mean()))

    """Agreement on the country, over the articles the pipeline actually attributed"""
    both = joined[joined.location_GID_0.notna() & joined.label_country.notna()]
    if len(both):
        agree = (both.label_country == both.location_GID_0).mean()
        print("\n  agree on country   {:>6.1%}  (n = {:,} both attributed)".format(
            agree, len(both)))
        disagreeing = both[both.label_country != both.location_GID_0]
        for row in disagreeing.head(5).itertuples():
            print("    {}  annotated {} ({})  pipeline {} ({})".format(
                row.sample_id, row.label_country, str(row.label_location)[:22],
                row.location_GID_0, str(row.location_closest)[:22]))
        if len(disagreeing) > 5:
            print("    ... and {:,} more".format(len(disagreeing) - 5))

    mapped = joined.location_GID_0.notna()
    print("\n  pipeline attributed, annotation did not  {:,}".format(
        int((mapped & joined.label_country.isna()).sum())))
    print("  annotation attributed, pipeline did not  {:,}".format(
        int((~mapped & joined.label_country.notna()).sum())))

    """Country names the model used that resolve to no ISO3 code"""
    unresolved = pd.Series([
        name for name, code in zip(merged["country_name"].fillna(""), merged["label_country"])
        if name and pd.isna(code)
    ])
    if len(unresolved):
        print("\n  {:,} country name(s) resolved to no ISO3 code "
              "(oceans and continents expected, misspellings not):".format(len(unresolved)))
        for name, count in unresolved.value_counts().head(10).items():
            print("    {:<38} {:,}".format(str(name)[:38], count))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="send unannotated articles to the Batch API")
    submit.add_argument("--smoke", type=int, metavar="N",
                        help="submit only the first N articles, to check the prompt first")
    submit.set_defaults(function=cmd_submit)

    status = subparsers.add_parser("status", help="print the state of every submitted batch")
    status.set_defaults(function=cmd_status)

    collect = subparsers.add_parser("collect", help="download finished batches into the raw file")
    collect.set_defaults(function=cmd_collect)

    merge = subparsers.add_parser("merge", help="write the annotated CSV from the raw file")
    merge.set_defaults(function=cmd_merge)

    for subparser in (submit, status, collect, merge):
        subparser.add_argument("--sample", choices=list(SAMPLES), default="main",
                               help="the sample to annotate: main (default) or continent, "
                                    "the continent-stratified sample of the classified articles")

    args = parser.parse_args()
    use_sample(args.sample)
    args.function(args)


if __name__ == "__main__":
    main()
