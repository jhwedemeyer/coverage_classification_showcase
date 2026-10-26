"""
Draw the blinded, stratified test sample used to validate the classification pipeline.

The sample is stratified by classifier outcome:
- 300 articles per impact type from the classified articles (n = 1,500), for precision
- 1,500 articles from the unclassified articles, for recall

Duplicates and syndicated copies are removed while drawing. The script writes an annotation
sheet without the classifier's output, plus a separate key recording what the pipeline decided
for each article: the impact types, the selected location and its country. The location
selection reuses gen_coverage_days.select_location(), so the validation tests exactly what the
final dataset is built on.

Two further samples are drawn from it:
- a pilot subset of 300 articles in the same proportions (plus 30 African articles), and
- a continent sample of 250 classified articles per continent, for evaluate_geo_bias.py.

    python src/generate_test_sample.py            # draw the sample, then add locations
    python src/generate_test_sample.py locations  # add locations to an existing sample
    python src/generate_test_sample.py small      # pilot subset only
    python src/generate_test_sample.py continent  # continent sample only

Existing annotated sheets are never overwritten. Run from the repository root.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import bag_of_words as bow
from news_data_filter import GeneralFilter, NewsFilter
from config import NEWS_DB_PATH as DATABASE

# the location selection and the geocoding it maps through are gen_coverage_days.py's own, so
# that the validation cannot drift from the attribution the panel is built on
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gen_coverage_days import GEOCODED, load_geocoding, location_mapping, select_location

BAG_OF_WORDS = 'data/bag_of_words.yml'
OUTDIR = 'data/testsample/'

ANNOTATION_FILE = OUTDIR + 'test_sample_annotation.csv'
KEY_FILE = OUTDIR + 'test_sample_key.csv'
METADATA_FILE = OUTDIR + 'test_sample_metadata.json'

# The pilot subset of the sample, in the same format
SMALL_ANNOTATION_FILE = OUTDIR + 'test_sample_small_annotation.csv'
SMALL_KEY_FILE = OUTDIR + 'test_sample_small_key.csv'

# The continent sample, drawn from the classified articles alone, in the same format
CONTINENT_ANNOTATION_FILE = OUTDIR + 'test_sample_continent_annotation.csv'
CONTINENT_KEY_FILE = OUTDIR + 'test_sample_continent_key.csv'

# ISO3 -> continent, the table the panel's own Continent column is from
COUNTRIES = '../geo_data/countries-and-meta.pkl'

# Hazard classes of the panel, identical to DISASTER_TYPES in gen_coverage_days.py.
EVENTS = ["Flood", "Wildfire", "Heat", "Storm", "Cold"]

# Output of classify_locations.py, ranked by ranking_locations.py
LOCATIONS = ('data/eventcoverage/eventcoverage_{dtype}_localised_ranked.parquet')

# The classifier searches the whole corpus; the sample is restricted to the
# analysis period here. Set both to None to sample from the full corpus.
START_DATE = pd.to_datetime("1990-01-01")
END_DATE = pd.to_datetime("2022-12-31")

# Journals the panel is not built from, dropped here so that the sample is drawn
# from the same corpus. gen_coverage_days.py drops zeit_offline, the digitised
# print archive of Die Zeit, which duplicates the online edition.
EXCLUDED_JOURNALS = ["zeit_offline"]

N_PER_CLASS = 300          # articles drawn per classified stratum (precision)
N_UNCLASSIFIED = 1500      # articles drawn from the unclassified stratum (recall)

# Pilot sample, n = 300 in total, a tenth of the full sample and drawn in the
# same proportions, so that it is a stratified random sample of the same corpus
N_PER_CLASS_SMALL = 30
N_UNCLASSIFIED_SMALL = 150
# African articles of the continent sample added to the pilot on top of it
N_AFRICA_SMALL = 30

# Continent sample, n = 1,500 classified articles, the same number from every
# continent the pipeline attributes articles to. Antarctica is left out, the
# pipeline attributes no article to it.
CONTINENTS = ["AF", "AS", "EU", "NA", "OC", "SA"]
N_PER_CONTINENT = 250

SEED = 20240210

DRAW_CHUNK = 512           # candidates fetched from the database per round
TEXT_PREFIX = 400          # normalised characters used to detect syndicated copies

"""Load bag of words"""

def load_bag_of_words(path=BAG_OF_WORDS):
    return bow.load(path)


"""Full text search: ids of the articles the classifier assigns to each hazard class"""

def hazard_ids(con, bag_of_words):
    ids = {}
    for event in EVENTS:
        # the same query classify_impact_type.py runs, prefix keywords and all, so that the
        # strata are exactly the classifier's own output
        event_bag_of_words = bow.fts_query(bag_of_words[event])
        # news_fts is an external content table with content_rowid='id',
        # so the fts rowid is the primary key of *news*.
        rows = con.execute(
            "SELECT rowid FROM news_fts WHERE news_fts MATCH ?", (event_bag_of_words,)
        ).fetchall()
        ids[event] = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        print("Classified as {}: {:,} articles".format(event, len(ids[event])))
    return ids


def corpus_ids(con):
    # served from the covering index idx_news_journal, no scan of the text column.
    # streamed from the cursor, materialising all ids as tuples needs several GB.
    cursor = con.execute("SELECT id FROM news")
    return np.fromiter((r[0] for r in cursor), dtype=np.int64)


def fetch_articles(con, ids):
    columns = ["id", "title", "date", "link", "journal", "text"]
    records = []
    for start in range(0, len(ids), DRAW_CHUNK):
        chunk = ids[start:start + DRAW_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        records += con.execute(
            "SELECT " + ", ".join(columns) + " FROM news WHERE id IN (" + placeholders + ")",
            [int(i) for i in chunk],
        ).fetchall()
    return pd.DataFrame(records, columns=columns)


"""Corpus filters"""


def apply_corpus_filters(articles):
    """Restrict the candidates to the corpus the classifier is run on.

    NewsFilter is the same filter that NewsData.search() applies in
    classify_impact_type.py, so the sample is drawn from exactly the corpus the
    classifier sees. The date window and the excluded journals are not part of
    NewsFilter and are applied on top of it, as gen_coverage_days.py does.
    """
    articles = NewsFilter.apply(articles)
    articles = articles[~articles.journal.isin(EXCLUDED_JOURNALS)]
    if START_DATE is not None:
        articles = articles[articles.date >= START_DATE]
    if END_DATE is not None:
        articles = articles[articles.date <= END_DATE]
    return articles


"""Duplicate and syndicated content"""


def fingerprints(text):
    """Return an exact and a prefix fingerprint of the normalised article text.

    The exact fingerprint removes verbatim duplicates, the prefix fingerprint
    removes syndicated copies that differ only in their trailing paragraphs.
    """
    normalised = re.sub(r"\W+", " ", str(text).lower()).strip()
    if not normalised:
        return None, None
    exact = hashlib.sha1(normalised.encode("utf-8")).hexdigest()
    prefix = hashlib.sha1(normalised[:TEXT_PREFIX].encode("utf-8")).hexdigest()
    return exact, prefix


"""Stratified sampling"""


def draw_sample(con, pool, n_target, rng, seen, stratum):
    """Draw *n_target* articles from *pool* by walking a random permutation.

    Candidates that are not part of the filtered corpus, that carry no text or
    that duplicate an already accepted article are skipped and replaced by the
    next candidate of the permutation, which keeps the accepted articles a
    simple random sample of the deduplicated stratum. *seen* is shared between
    strata so that no text enters the annotation set twice.
    """
    permutation = rng.permutation(pool)
    accepted, examined = [], 0
    rejected = {"not_in_corpus": 0, "no_text": 0, "duplicate": 0, "missing": 0}
    cursor = 0

    while len(accepted) < n_target and cursor < len(permutation):
        candidates = permutation[cursor:cursor + DRAW_CHUNK]
        cursor += len(candidates)

        articles = fetch_articles(con, candidates)
        in_corpus = set(apply_corpus_filters(articles).id) if len(articles) else set()
        articles = articles.set_index("id")

        for article_id in candidates:
            if len(accepted) >= n_target:
                break
            examined += 1
            article_id = int(article_id)
            if article_id not in articles.index:
                rejected["missing"] += 1
                continue
            if article_id not in in_corpus:
                rejected["not_in_corpus"] += 1
                continue
            article = articles.loc[article_id]
            exact, prefix = fingerprints(article.text)
            if exact is None:
                rejected["no_text"] += 1
                continue
            if exact in seen or prefix in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(exact)
            seen.add(prefix)
            accepted.append({"id": article_id, "stratum": stratum, **article.to_dict()})

    sample = pd.DataFrame(accepted)
    statistics = {
        "stratum": stratum,
        "population_raw": int(len(pool)),
        "n_drawn": int(len(sample)),
        "n_examined": int(examined),
        "rejected": rejected,
        # share of the raw stratum that survives corpus filtering and deduplication
        "retention_rate": float(len(sample) / examined) if examined else float("nan"),
        "population_effective": float(len(pool) * len(sample) / examined) if examined else float("nan"),
    }
    print(
        "Stratum {}: drew {:,} of {:,} examined candidates "
        "(not in corpus {:,}, no text {:,}, duplicate {:,})".format(
            stratum, len(sample), examined,
            rejected["not_in_corpus"], rejected["no_text"], rejected["duplicate"],
        )
    )
    if len(sample) < n_target:
        print("WARNING: stratum {} exhausted before reaching n = {}".format(stratum, n_target))
    return sample, statistics


"""The pilot subset of the sample"""


def small_sample_ids(sample):
    """The sample ids of the pilot subset, drawn per stratum from the full sample.

    Drawn stratum by stratum rather than as one simple random sample of the
    3,000, so that the pilot holds exactly the intended number of articles of
    every stratum and its strata can be weighted by the same population sizes
    as those of the full sample. The subset is nested: the articles it holds
    are the same articles, under the same sample ids, so the labels written for
    the pilot carry over to the full annotation set unchanged.

    Its own generator, so that which articles the pilot holds does not depend
    on how many candidates the draw from the database had to walk through.
    """
    rng = np.random.default_rng(SEED + 1)
    targets = {"unclassified": N_UNCLASSIFIED_SMALL}
    targets.update({"classified:" + event: N_PER_CLASS_SMALL for event in EVENTS})

    chosen = []
    for stratum, n_target in targets.items():
        ids = sample.sample_id[sample.stratum == stratum].to_numpy()
        if len(ids) < n_target:
            print("WARNING: stratum {} holds only {:,} of the {:,} articles the "
                  "pilot sample asks for".format(stratum, len(ids), n_target))
            n_target = len(ids)
        chosen += list(rng.choice(ids, size=n_target, replace=False))
    return set(chosen)


def small_africa_ids(pilot_links):
    """The sample ids of the African articles of the continent sample added to the pilot.

    Drawn from the continent:AF stratum of the continent sample, leaving out the
    articles the pilot already holds. Empty where the continent sample has not
    been drawn yet. Its own generator, like the pilot's.
    """
    if not os.path.exists(CONTINENT_KEY_FILE):
        print("WARNING: no continent sample yet, the pilot holds no additional African articles")
        return set()
    continent = pd.read_csv(CONTINENT_KEY_FILE)
    ids = continent.sample_id[(continent.stratum == "continent:AF")
                              & ~continent.link.isin(pilot_links)].to_numpy()
    n_target = N_AFRICA_SMALL
    if len(ids) < n_target:
        print("WARNING: the continent sample holds only {:,} African articles outside the "
              "pilot, of the {:,} asked for".format(len(ids), n_target))
        n_target = len(ids)
    rng = np.random.default_rng(SEED + 3)
    return set(rng.choice(ids, size=n_target, replace=False))


def pilot_key(key, ids):
    """The rows of the main key and of the continent key the pilot holds."""
    frames = [key[key.sample_id.isin(ids)]]
    if os.path.exists(CONTINENT_KEY_FILE):
        continent = pd.read_csv(CONTINENT_KEY_FILE)
        frames.append(continent[continent.sample_id.isin(ids)])
    return pd.concat(frames, ignore_index=True)


def small_metadata(small_key):
    return {
        "annotation": SMALL_ANNOTATION_FILE,
        "key": SMALL_KEY_FILE,
        "n_per_class": N_PER_CLASS_SMALL,
        "n_unclassified": N_UNCLASSIFIED_SMALL,
        "n_africa": int((small_key.stratum == "continent:AF").sum()),
        "n_annotation_set": int(len(small_key)),
        "strata": {stratum: int(count) for stratum, count
                   in small_key.stratum.value_counts().items()},
    }


def write_small_sample(annotation, key, ids):
    """Write the pilot subset of an annotation sheet and its key.

    Both are written by filtering the full files, so the pilot carries the same
    columns in the same order. The African articles of the continent sample,
    small_africa_ids(), are added from its files and the sheet is shuffled once
    more, so that they do not stand together at its end and it stays as blind
    as the sheets it is taken from.
    """
    small_annotation = annotation[annotation.sample_id.isin(ids)]
    africa = small_africa_ids(set(key.link[key.sample_id.isin(ids)]))
    if africa:
        continent = pd.read_csv(CONTINENT_ANNOTATION_FILE, keep_default_na=False, dtype=str)
        small_annotation = pd.concat(
            [small_annotation, continent[continent.sample_id.isin(africa)]], ignore_index=True)
        small_annotation = small_annotation.sample(frac=1.0, random_state=SEED + 3)
    small_key = pilot_key(key, set(ids) | africa)
    small_annotation.to_csv(SMALL_ANNOTATION_FILE, index=False)
    small_key.to_csv(SMALL_KEY_FILE, index=False)
    print("Wrote the pilot sample of {:,} articles to {} and {}".format(
        len(small_annotation), SMALL_ANNOTATION_FILE, SMALL_KEY_FILE))
    return small_key


"""Locations: the place the pipeline reads an article as reporting on"""


def selected_location(locations, ranks, same_sentence, distances_words, distances_chars,
                      keywords, mapping):
    """The location gen_coverage_days.py attributes the article by, with what the key records of it.

    The choice itself is gen_coverage_days.select_location(): the highest ranked
    location that names a country, the locations sharing a sentence with a
    keyword of the hazard ranked first. `location_rank` records how far down the
    ranking the selected location sat, 1 being the top of it, and
    `location_n_located` how many locations were ranked at all.
    """
    i, gid_0, position, n_ranked = select_location(locations, ranks, mapping)
    if i is None:
        return {"location_closest": None, "location_rank": None,
                "location_same_sentence": None, "location_distance_words": None,
                "location_distance_chars": None, "location_keyword": None,
                "location_GID_0": None, "location_n_located": n_ranked}
    return {
        "location_closest": locations[i],
        "location_rank": position,
        "location_same_sentence": bool(same_sentence[i]) if i < len(same_sentence) else None,
        "location_distance_words": int(distances_words[i]),
        "location_distance_chars": int(distances_chars[i]),
        "location_keyword": keywords[i] if i < len(keywords) else None,
        "location_GID_0": gid_0,
        "location_n_located": n_ranked,
    }


def ranking_key(record):
    """How a selected location sorts against one selected from another hazard's file.

    The ranks of two files are positions within different rankings and are not
    comparable, so the key they are built from is compared instead, the one
    ranking_locations.rank() sorts by: sharing a sentence with a keyword first,
    then the word distance, then the character distance.
    """
    return (not record["location_same_sentence"], record["location_distance_words"],
            record["location_distance_chars"])


def location_files():
    """The files of ranking_locations.py that are there, by hazard.

    Empty before the location stage of the pipeline has been run, and short of
    a hazard while it is still running, which the location stage here treats as
    "no locations yet" rather than as an error.
    """
    paths = {event: LOCATIONS.format(dtype=event.lower()) for event in EVENTS}
    return {event: path for event, path in paths.items() if os.path.exists(path)}


LOCATION_FIELDS = ["link", "flair_locations", "location_rank", "same_sentence",
                   "keyword_distance_words", "keyword_distance_chars", "nearest_keyword"]


def as_list(value):
    """The entries of one of the list columns of ranking_locations.py.

    Parquet reads those columns back as numpy arrays rather than as the lists
    they were written from, and leaves them null where the article carries no
    location at all, so the columns are normalised here once.
    """
    return [] if value is None else list(value)


def tagged_locations(links, mapping, files):
    """Locations of the sampled articles, read from the output of ranking_locations.py.

    The files hold no article id, so they are joined on the link, which is
    unique within each of them. An article that carries the keywords of several
    hazards appears in several files with the same tagged locations but
    different distances; the panel attributes it once per hazard, the key keeps
    the one selection that ranks highest over all of them, see ranking_key(),
    together with the hazard whose keyword it stands next to.
    """
    tagged = {}
    for event, path in files.items():
        data = pd.read_parquet(path, columns=LOCATION_FIELDS)
        data = data[data.link.isin(links)]
        for article in data.itertuples():
            locations = as_list(article.flair_locations)
            record = {
                "location_tagged": "|".join(locations),
                "location_n_tagged": len(locations),
            }
            closest = selected_location(
                locations,
                as_list(article.location_rank),
                as_list(article.same_sentence),
                as_list(article.keyword_distance_words),
                as_list(article.keyword_distance_chars),
                as_list(article.nearest_keyword),
                mapping,
            )
            record.update(closest)
            if closest["location_closest"] is not None:
                record["location_dtype"] = event

            previous = tagged.get(article.link)
            if previous is None or (
                closest["location_closest"] is not None and (
                    previous["location_closest"] is None
                    or ranking_key(record) < ranking_key(previous)
                )
            ):
                tagged[article.link] = record
        print("Read the tagged locations of {:,} {} articles".format(len(data), event))
    return tagged


"""Locations: the country the pipeline maps that place to"""


def location_details(record, geocoding):
    """What the geocoding says about the selected location, for the key.

    The country itself, location_GID_0, is already on the record: it is the one
    select_location() mapped the location to while selecting it, so it cannot
    differ from the panel's. What is added here is only what the geocoding
    records of the location, the feature type, the country name and the two
    letter code, and the status. select_location() has skipped the locations
    that name no country, so an article is left without one only when none of
    its locations names any.
    """
    location = record.get("location_closest")
    if location is None or pd.isna(location):
        return {"location_type": None, "location_country": None,
                "location_country_code": None, "location_GID_0": None,
                "location_status": "no_country" if record.get("location_n_located")
                else "no_location"}

    entry = geocoding.get(str(location).lower(), {})
    return {
        "location_type": entry.get("location_type"),
        "location_country": entry.get("country"),
        "location_country_code": entry.get("country_code"),
        "location_status": "mapped",
    }


LOCATION_COLUMNS = [
    "location_tagged", "location_n_tagged", "location_n_located", "location_closest",
    "location_rank", "location_same_sentence", "location_distance_words",
    "location_distance_chars",
    "location_keyword", "location_dtype", "location_type", "location_country",
    "location_country_code", "location_GID_0", "location_status",
]


def add_locations(key):
    """Add the location the pipeline reads out of every sampled article, and the
    country it maps it to, to the key.

    Only the articles of the classified strata carry locations: the pipeline
    tags the articles the dictionary classifier returned, so the articles of the
    unclassified stratum are never seen by it and are left empty here.

    The location stage of the pipeline runs after the sample is drawn, so the
    files it writes need not be there yet. Where they are not, the columns are
    left empty and the sample is written all the same; rerun the stage on its
    own once the locations are there:

        python src/generate_test_sample.py locations
    """
    classified = key[key.stratum != "unclassified"]
    files = location_files()
    if not files or not os.path.exists(GEOCODED):
        print("No locations yet ({}), leaving the location columns empty".format(
            GEOCODED if files else LOCATIONS.format(dtype="*")))
        empty = pd.DataFrame(None, index=key.index, columns=LOCATION_COLUMNS)
        key = key.drop(columns=LOCATION_COLUMNS, errors="ignore")
        return pd.concat([key, empty], axis=1), {}

    if len(files) < len(EVENTS):
        print("WARNING: no location file for {}, those articles are left empty".format(
            ", ".join(event for event in EVENTS if event not in files)))

    geocoding = load_geocoding()
    tagged = tagged_locations(set(classified.link), location_mapping(geocoding), files)
    geocoding = geocoding.to_dict("index")
    missing = set(classified.link) - set(tagged)
    if missing:
        print("WARNING: {:,} classified articles are not in the location files".format(
            len(missing)))

    records = []
    for link, stratum in zip(key.link, key.stratum):
        if stratum == "unclassified":
            # never tagged, which is not the same as tagged without a location
            records.append({"location_status": "not_classified"})
            continue
        record = dict(tagged.get(link, {}))
        record.update(location_details(record, geocoding))
        records.append(record)

    locations = pd.DataFrame(records, index=key.index).reindex(columns=LOCATION_COLUMNS)
    key = key.drop(columns=LOCATION_COLUMNS, errors="ignore")
    key = pd.concat([key, locations], axis=1)

    status = locations.loc[classified.index, "location_status"].value_counts(dropna=False)
    print("Locations of the {:,} classified articles:".format(len(classified)))
    for name, count in status.items():
        print("  {:<20} {:>5,} ({:.1%})".format(str(name), count, count / len(classified)))
    return key, {str(name): int(count) for name, count in status.items()}


LOCATION_LABELS = ["label_location", "label_country", "label_no_location",
                   "label_location_unclear"]


def add_location_labels(annotation):
    """Add the empty columns the annotators fill in for the location.

    label_location is the place the article reports the hazard as occurring in,
    label_country the country that place lies in. Both are needed: the first
    tests which of the tagged locations the closest-keyword rule should have
    picked, the second the country the pipeline maps it to.
    """
    annotation = annotation.drop(columns=LOCATION_LABELS, errors="ignore")
    for label in LOCATION_LABELS:
        annotation[label] = ""
    if "notes" in annotation:  # keep the free text column last
        annotation["notes"] = annotation.pop("notes")
    return annotation


def locations_command():
    """Run the location stage on an already drawn sample."""
    key = pd.read_csv(KEY_FILE)
    key, status = add_locations(key)
    key.to_csv(KEY_FILE, index=False)

    # keep_default_na, so that articles whose title or text is the string "NULL"
    # survive the round trip through the annotation sheet unchanged
    annotation = pd.read_csv(ANNOTATION_FILE, keep_default_na=False, dtype=str)
    add_location_labels(annotation).to_csv(ANNOTATION_FILE, index=False)

    update_small_sample(key)

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE) as stream:
            metadata = json.load(stream)
        metadata["locations"] = location_metadata(status)
        with open(METADATA_FILE, "w") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)

    print("Added the locations of {:,} articles to {} and {}".format(
        len(key), KEY_FILE, ANNOTATION_FILE))


def update_small_sample(key):
    """Bring the pilot files up to the columns the location stage just added.

    The pilot annotation sheet may already be filled in by the time the
    locations are there, so it is not rewritten from the full sheet: only the
    empty location columns are added to it, as they are to the full one. Its
    key holds nothing an annotator writes and is rewritten from the full key.
    """
    if not os.path.exists(SMALL_ANNOTATION_FILE):
        return
    annotation = pd.read_csv(SMALL_ANNOTATION_FILE, keep_default_na=False, dtype=str)
    add_location_labels(annotation).to_csv(SMALL_ANNOTATION_FILE, index=False)

    ids = set(annotation.sample_id)
    pilot_key(key, ids).to_csv(SMALL_KEY_FILE, index=False)
    print("Added the locations of {:,} articles to {} and {}".format(
        len(ids), SMALL_KEY_FILE, SMALL_ANNOTATION_FILE))


def small_command():
    """Take the pilot subset out of an already drawn sample.

    Reads the annotation sheet and the key rather than the database, so the
    sample they hold is left as it is -- drawing it again would draw a
    different one, and the annotations written for it so far would no longer
    belong to any sample. Rerunning this writes the same subset of the same
    sample, and refuses to overwrite a pilot sheet that is already filled in.
    """
    key = pd.read_csv(KEY_FILE)
    annotation = pd.read_csv(ANNOTATION_FILE, keep_default_na=False, dtype=str)

    if is_annotated(SMALL_ANNOTATION_FILE):
        print("{} is already annotated, leaving it as it is".format(SMALL_ANNOTATION_FILE))
        return

    small_key = write_small_sample(annotation, key, small_sample_ids(key))
    update_metadata("small_sample", small_metadata(small_key))


def is_annotated(path):
    """Whether the annotation sheet at *path* exists and holds any label."""
    if not os.path.exists(path):
        return False
    sheet = pd.read_csv(path, keep_default_na=False, dtype=str)
    labels = [column for column in sheet.columns if column.startswith("label_")]
    return bool(sheet[labels].replace("", pd.NA).notna().any().any())


def update_metadata(field, value):
    """Set one field of the metadata of an already drawn sample, leaving the rest."""
    if not os.path.exists(METADATA_FILE):
        return
    with open(METADATA_FILE) as stream:
        metadata = json.load(stream)
    metadata[field] = value
    with open(METADATA_FILE, "w") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)


"""The continent sample"""


def continent_lookup():
    """ISO3 -> two-letter continent code, as evaluate_geo_bias.continent_lookup()."""
    countries = pd.read_pickle(COUNTRIES)
    return dict(zip(countries["ISO3"], countries["Continent"]))


def classified_frame():
    """Every classified article of the corpus the panel is built from, with its text.

    Read from the ranked location files, which hold the articles the classifier
    returned for each hazard, so no database is needed. The journals and the
    date window are restricted as for the main sample. An article carrying the
    keywords of several hazards is in several files and kept once, with every
    hazard it was classified as in `classifier_output`.
    """
    frames = []
    for event, path in location_files().items():
        data = pd.read_parquet(path, columns=["link", "title", "text", "date", "journal"])
        data = data[~data.journal.isin(EXCLUDED_JOURNALS)]
        if START_DATE is not None:
            data = data[data.date >= START_DATE]
        if END_DATE is not None:
            data = data[data.date <= END_DATE]
        frames.append(data.assign(event=event))
    frame = pd.concat(frames, ignore_index=True)
    events = frame.groupby("link", sort=False).event.agg(
        lambda found: "|".join(event for event in EVENTS if event in set(found)))
    frame = frame.drop_duplicates("link").drop(columns="event").set_index("link")
    frame["classifier_output"] = events
    return frame


def draw_from_frame(frame, n_target, rng, seen, stratum):
    """Draw *n_target* articles from *frame* by walking a random permutation.

    The counterpart of draw_sample() for articles that are already in memory:
    articles without text or duplicating an already accepted one are skipped
    and replaced by the next of the permutation, so the accepted articles are a
    simple random sample of the deduplicated stratum.
    """
    accepted, examined = [], 0
    rejected = {"no_text": 0, "duplicate": 0}
    for link in rng.permutation(frame.index.to_numpy()):
        if len(accepted) >= n_target:
            break
        examined += 1
        article = frame.loc[link]
        exact, prefix = fingerprints(article.text)
        if exact is None:
            rejected["no_text"] += 1
            continue
        if exact in seen or prefix in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(exact)
        seen.add(prefix)
        accepted.append({"link": link, "stratum": stratum, **article.to_dict()})

    sample = pd.DataFrame(accepted)
    statistics = {
        "stratum": stratum,
        "population_raw": int(len(frame)),
        "n_drawn": int(len(sample)),
        "n_examined": int(examined),
        "rejected": rejected,
        "retention_rate": float(len(sample) / examined) if examined else float("nan"),
        "population_effective": float(len(frame) * len(sample) / examined) if examined else float("nan"),
    }
    print("Stratum {}: drew {:,} of {:,} examined candidates out of {:,} "
          "(no text {:,}, duplicate {:,})".format(
              stratum, len(sample), examined, len(frame),
              rejected["no_text"], rejected["duplicate"]))
    if len(sample) < n_target:
        print("WARNING: stratum {} exhausted before reaching n = {}".format(stratum, n_target))
    return sample, statistics


def continent_command():
    """Draw the continent sample: N_PER_CONTINENT classified articles per continent.

    Every classified article is attributed as the panel attributes it, by
    tagged_locations(), and the ones attributed to a country of one of
    CONTINENTS make up the stratum of that continent. Articles attributed to no
    country cannot be placed on a continent and are outside the frame.
    """
    if is_annotated(CONTINENT_ANNOTATION_FILE):
        print("{} is already annotated, leaving it as it is".format(CONTINENT_ANNOTATION_FILE))
        return
    files = location_files()
    if len(files) < len(EVENTS) or not os.path.exists(GEOCODED):
        print("The continent sample needs the location files of all hazards and {}, "
              "not drawing it".format(GEOCODED))
        return
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    frame = classified_frame()
    geocoding = load_geocoding()
    tagged = tagged_locations(set(frame.index), location_mapping(geocoding), files)
    geocoding = geocoding.to_dict("index")

    records = []
    for link in frame.index:
        record = dict(tagged.get(link, {}))
        record.update(location_details(record, geocoding))
        records.append(record)
    locations = pd.DataFrame(records, index=frame.index).reindex(columns=LOCATION_COLUMNS)
    frame = frame.join(locations)
    frame["continent"] = frame.location_GID_0.map(continent_lookup())
    print("Classified articles: {:,}, attributed to a continent: {:,}".format(
        len(frame), frame.continent.isin(CONTINENTS).sum()))

    """Draw every continent from its own generator stream, sharing the
    duplicate check so that no text enters the sheet twice"""
    rng = np.random.default_rng(SEED + 2)
    seen = set()
    samples, statistics = [], []
    for continent in CONTINENTS:
        sample, stats = draw_from_frame(
            frame[frame.continent == continent], N_PER_CONTINENT, rng, seen,
            "continent:" + continent)
        samples.append(sample)
        statistics.append(stats)
    sample = pd.concat(samples, ignore_index=True)
    sample["date"] = pd.to_datetime(sample.date).dt.date

    """Blind the annotation set: shuffle and strip every trace of the strata"""
    sample = sample.sample(frac=1.0, random_state=SEED + 2).reset_index(drop=True)
    sample["sample_id"] = ["C{:04d}".format(i + 1) for i in range(len(sample))]

    annotation = sample[["sample_id", "date", "journal", "title", "text", "link"]].copy()
    for event in EVENTS:
        annotation["label_" + event.lower()] = ""
    annotation["label_no_hazard"] = ""
    annotation["label_unclear"] = ""
    annotation["notes"] = ""
    annotation = add_location_labels(annotation)
    annotation.to_csv(CONTINENT_ANNOTATION_FILE, index=False)

    """The article of the main sample each one is, if any, so that its labels carry over"""
    main_ids = pd.read_csv(KEY_FILE).set_index("link").sample_id if os.path.exists(KEY_FILE) \
        else pd.Series(dtype=str)
    sample["sample_id_main"] = sample.link.map(main_ids)
    print("{:,} articles of the continent sample are in the main sample".format(
        sample.sample_id_main.notna().sum()))

    key = sample[["sample_id", "stratum", "continent", "classifier_output", "date",
                  "journal", "link", "sample_id_main"] + LOCATION_COLUMNS]
    key.to_csv(CONTINENT_KEY_FILE, index=False)

    update_metadata("continent_sample", {
        "annotation": CONTINENT_ANNOTATION_FILE,
        "key": CONTINENT_KEY_FILE,
        "seed": SEED + 2,
        "continents": CONTINENTS,
        "n_per_continent": N_PER_CONTINENT,
        "classified_raw": int(len(frame)),
        "attributed_raw_per_continent": {
            str(c): int(n) for c, n in frame.continent.value_counts().items()},
        "n_annotation_set": int(len(sample)),
        "n_in_main_sample": int(sample.sample_id_main.notna().sum()),
        "strata": statistics,
    })
    print("Wrote the continent sample of {:,} articles to {} and {}".format(
        len(sample), CONTINENT_ANNOTATION_FILE, CONTINENT_KEY_FILE))


def location_metadata(status):
    return {
        "locations": LOCATIONS,
        "geocoding": GEOCODED,
        "status": status,
    }


def main():
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)

    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:" + DATABASE + "?mode=ro", uri=True)

    bag_of_words = load_bag_of_words()

    """Define the strata by the outcome of the classifier"""
    classified = hazard_ids(con, bag_of_words)
    classified_union = np.unique(np.concatenate([classified[event] for event in EVENTS]))
    corpus = corpus_ids(con)
    unclassified = np.setdiff1d(corpus, classified_union, assume_unique=False)
    print("Corpus: {:,} articles, classified: {:,}, unclassified: {:,}".format(
        len(corpus), len(classified_union), len(unclassified)))

    """Draw the samples, the unclassified stratum first so that its articles
    are not blocked by duplicates of the much smaller classified strata"""
    seen = set()
    samples, statistics = [], []

    sample, stats = draw_sample(
        con, unclassified, N_UNCLASSIFIED, rng, seen, "unclassified")
    samples.append(sample)
    statistics.append(stats)

    for event in EVENTS:
        sample, stats = draw_sample(
            con, classified[event], N_PER_CLASS, rng, seen, "classified:" + event)
        samples.append(sample)
        statistics.append(stats)

    sample = pd.concat(samples, ignore_index=True)
    # the accepted rows carry the raw date strings of the database
    sample["date"] = GeneralFilter.apply(sample)["date"].dt.date

    """An article drawn for one hazard class can also carry the keywords of
    another one. The classifier outcome of every drawn article is recorded so
    that the class specific precision can be computed on the full 300 articles."""
    membership = {event: set(classified[event].tolist()) for event in EVENTS}
    classifier_output = {
        article_id: [event for event in EVENTS if article_id in membership[event]]
        for article_id in sample.id
    }
    sample["classifier_output"] = sample.id.map(lambda i: "|".join(classifier_output[i]))

    """Blind the annotation set: shuffle and strip every trace of the strata"""
    sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample["sample_id"] = ["S{:04d}".format(i + 1) for i in range(len(sample))]

    annotation = sample[["sample_id", "date", "journal", "title", "text", "link"]].copy()
    for event in EVENTS:
        annotation["label_" + event.lower()] = ""
    annotation["label_no_hazard"] = ""
    annotation["label_unclear"] = ""
    annotation["notes"] = ""
    annotation = add_location_labels(annotation)
    annotation.to_csv(ANNOTATION_FILE, index=False)

    key = sample[["sample_id", "id", "stratum", "classifier_output", "date", "journal", "link"]]
    key = key.rename(columns={"id": "article_id"})

    """The location the pipeline reads out of the article and the country it
    maps it to, the counterpart of the location labels of the annotation sheet"""
    key, location_status = add_locations(key)
    key.to_csv(KEY_FILE, index=False)

    """Population sizes for the Horvitz-Thompson estimator"""
    metadata = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": DATABASE,
        "bag_of_words": BAG_OF_WORDS,
        "seed": SEED,
        "events": EVENTS,
        "start_date": str(START_DATE) if START_DATE is not None else None,
        "end_date": str(END_DATE) if END_DATE is not None else None,
        "excluded_journals": EXCLUDED_JOURNALS,
        "n_per_class": N_PER_CLASS,
        "n_unclassified": N_UNCLASSIFIED,
        "corpus_size_raw": int(len(corpus)),
        "classified_raw": int(len(classified_union)),
        "unclassified_raw": int(len(unclassified)),
        "classified_raw_per_event": {e: int(len(classified[e])) for e in EVENTS},
        "n_annotation_set": int(len(sample)),
        "strata": statistics,
        "locations": location_metadata(location_status),
    }
    with open(METADATA_FILE, "w") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    con.close()
    print("Wrote {:,} articles for annotation to {}".format(len(sample), OUTDIR))

    """The continent sample, once the locations of all hazards are there"""
    continent_command()

    """The pilot subset, small enough to be annotated by hand right away, after
    the continent sample it takes its additional African articles from"""
    small_key = write_small_sample(annotation, key, small_sample_ids(sample))
    update_metadata("small_sample", small_metadata(small_key))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "stage", nargs="?", default="sample",
        choices=["sample", "locations", "small", "continent"],
        help="sample draws the articles and runs the location stage on them, "
             "locations runs the location stage on an already drawn sample, "
             "small takes the pilot subset out of an already drawn sample, "
             "continent draws the continent sample of the classified articles")
    arguments = parser.parse_args()
    if arguments.stage == "locations":
        locations_command()
    elif arguments.stage == "small":
        small_command()
    elif arguments.stage == "continent":
        continent_command()
    else:
        main()
