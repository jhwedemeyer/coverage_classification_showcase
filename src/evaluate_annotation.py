"""
Evaluate the keyword classifier and the location attribution against the annotated test sample.

Compares the annotation sheet (AI or human) with the unblinding key from generate_test_sample.py
and reports precision, recall and F1 per impact type, plus the accuracy of the place and country
each article is attributed to.

    python src/evaluate_annotation.py                                   # AI annotation
    python src/evaluate_annotation.py --annotation <human sheet.csv>    # human annotation

Statistical approach:
- The sample is stratified by classifier outcome, and classified strata overlap (an article can
  carry keywords of several impact types). Estimates over several strata therefore use
  Horvitz-Thompson weights (1 / inclusion probability).
- Proportions within a single stratum get Wilson score intervals; weighted estimates get a
  stratified bootstrap interval.
- Precision is reported both strictly ("about" only) and leniently ("about" or "mention").

Writes validation_metrics.csv, validation_errors.csv and validation_hazard.tex (a LaTeX
tabular) to data/testsample/. Run from the repository root.
"""

import argparse
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ai_annotation
from ai_annotation import EVENTS, to_iso3

TESTSAMPLE = ai_annotation.TESTSAMPLE
ANNOTATION_AI = ai_annotation.OUTFILE
KEY = ai_annotation.KEY
METADATA = TESTSAMPLE + "test_sample_metadata.json"

METRICS_FILE = TESTSAMPLE + "validation_metrics.csv"
ERRORS_FILE = TESTSAMPLE + "validation_errors.csv"
HAZARD_TABLE_FILE = TESTSAMPLE + "validation_hazard.tex"

# which grades count as the article reporting on the hazard
STRICTNESS = {"strict": ["about"], "lenient": ["about", "mention"]}

CONFIDENCE = 0.95          # nominal coverage of every interval reported here
BOOTSTRAP = 2000           # bootstrap replicates for the weighted estimates

# a normalised place name shorter than this is compared by equality only, so that "us" or "po"
# does not match every string it happens to sit inside
MIN_CONTAINMENT = 4


"""Intervals"""


def wilson(successes, total, confidence=CONFIDENCE):
    """Wilson score interval of a binomial proportion.

    Preferred over the normal approximation because several of the proportions here sit close to
    0 or 1 -- the miss rate in the unclassified stratum is a few percent -- where the Wald
    interval leaves the unit interval and undercovers. scipy computes it, in the uncorrected
    form; the point estimate stays the plain proportion, which is what the counts in the report
    are the numerator and denominator of.
    """
    if not total:
        return float("nan"), float("nan"), float("nan")
    low, high = stats.binomtest(int(successes), int(total)).proportion_ci(
        confidence_level=confidence, method="wilson")
    return successes / total, float(low), float(high)


def bootstrap(statistic, design, replicates, seed, confidence=CONFIDENCE):
    """Percentile interval of a weighted statistic, resampling within the strata.

    *statistic* is called with a row index array and returns a float, so it sees exactly the
    resampled articles. The strata go to scipy.stats.bootstrap as separate samples of row
    indices, and resampling each of them on its own is what makes the bootstrap stratified: every
    stratum stays at its drawn size, which is what the design fixes, and no row crosses from one
    stratum into another.

    The interval is read off the bootstrap distribution rather than taken from scipy's
    confidence_interval, so that the replicates the statistic is undefined on can be dropped
    instead of turning the percentile into a nan: a resample of a thin stratum can contain no
    annotated hazard at all, which leaves a recall without a denominator.
    """
    blocks = [np.flatnonzero(design["stratum_code"] == code)
              for code in np.unique(design["stratum_code"])]
    resampled = stats.bootstrap(
        blocks,
        lambda *strata: statistic(np.concatenate(strata)),
        n_resamples=replicates,
        confidence_level=confidence,
        method="percentile",
        vectorized=False,
        rng=np.random.default_rng(seed),
    ).bootstrap_distribution
    values = resampled[np.isfinite(resampled)]
    if not len(values):
        return float("nan"), float("nan")
    tail = 100 * (1 - confidence) / 2
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


"""The annotated sample"""


def load_metadata(path=METADATA):
    with open(path) as stream:
        return json.load(stream)


def load_sample(annotation_path, key_path=KEY):
    """The annotation sheet joined to the unblinding key, with the unclear articles dropped."""
    annotation = pd.read_csv(annotation_path)
    key = pd.read_csv(key_path)
    merged = annotation.merge(key.drop(columns=["date", "journal", "link"], errors="ignore"),
                              on="sample_id", how="inner")
    if len(merged) != len(annotation):
        print("WARNING: {:,} of {:,} annotated articles are not in the key".format(
            len(annotation) - len(merged), len(annotation)))

    graded = "grade_" + EVENTS[0].lower()
    filled = merged[graded] if graded in merged else merged["label_" + EVENTS[0].lower()]
    unannotated = int(filled.isna().sum())
    if unannotated == len(merged):
        sys.exit("{} carries no labels yet. Annotate it, or point --annotation at {}.".format(
            annotation_path, ANNOTATION_AI))
    if unannotated:
        print("{:,} of {:,} articles are not annotated and are left out".format(
            unannotated, len(merged)))
    merged = merged[filled.notna()]

    unclear = merged["label_unclear"].fillna(0).astype(float) > 0
    if unclear.any():
        print("{:,} article(s) marked unclear are left out".format(int(unclear.sum())))
    return merged[~unclear].reset_index(drop=True), int(unclear.sum())


def annotated_labels(sample, strictness):
    """Boolean matrix of what the annotation says each article reports on.

    Rebuilt from the grades rather than read from the label_ columns, so that both readings of a
    passing mention can be evaluated from one annotation. A sheet without grades -- the human one --
    falls back to its binary labels, which is the single reading it carries.
    """
    columns = []
    for event in EVENTS:
        grade = "grade_" + event.lower()
        if grade in sample:
            columns.append(sample[grade].isin(STRICTNESS[strictness]).to_numpy())
        else:
            columns.append((sample["label_" + event.lower()].fillna(0) > 0).to_numpy())
    return np.column_stack(columns)


def classifier_labels(sample):
    """Boolean matrix of what the keyword classifier assigned each article to."""
    output = sample["classifier_output"].fillna("")
    return np.column_stack([
        output.str.split("|").map(lambda events: event in events).to_numpy()
        for event in EVENTS
    ])


def design_weights(sample, metadata, population):
    """Horvitz-Thompson weight of every sampled article, and the strata it was drawn from."""
    field = "population_effective" if population == "effective" else "population_raw"
    sizes = {stratum["stratum"]: (stratum["n_drawn"], float(stratum[field]))
             for stratum in metadata["strata"]}

    rates = {}
    for event in EVENTS:
        drawn, size = sizes["classified:" + event]
        rates[event] = drawn / size
    drawn, size = sizes["unclassified"]
    unclassified_rate = drawn / size

    classified = classifier_labels(sample)
    probability = np.where(
        classified.any(axis=1),
        classified @ np.array([rates[event] for event in EVENTS]),
        unclassified_rate,
    )
    return 1.0 / probability


def build_design(sample, metadata, population, strictness):
    """Everything the estimators need, as plain arrays: weights, both label matrices, strata."""
    return {
        "weight": design_weights(sample, metadata, population),
        "classified": classifier_labels(sample),
        "annotated": annotated_labels(sample, strictness),
        "stratum_code": pd.Categorical(sample["stratum"]).codes,
        "stratum": sample["stratum"].to_numpy(),
    }


"""Estimates"""


def ratio(design, numerator, denominator):
    """A weighted proportion, as a function of a row index array, for the bootstrap."""
    weight = design["weight"]

    def statistic(rows):
        bottom = float(weight[rows] @ denominator[rows])
        if bottom <= 0:
            return float("nan")
        return float(weight[rows] @ numerator[rows]) / bottom

    return statistic


def hazard_metrics(design, replicates, seed, strictness):
    """Precision, recall and F1 of the classifier, per hazard and over all five together.

    Precision of a single hazard is measured inside its own stratum, which is a simple random
    sample of the articles the classifier assigns to that hazard, so it needs no weighting.
    Precision over all hazards together spans the five overlapping pools and is weighted, as is
    every recall: an article the classifier read as a storm and the annotation reads as a flood is
    a miss for flood, and it sits in the storm stratum, not the unclassified one.
    """
    rows = []
    all_rows = np.arange(len(design["weight"]))

    for index, event in enumerate(EVENTS + ["any"]):
        if event == "any":
            classified = design["classified"].any(axis=1)
            annotated = design["annotated"].any(axis=1)
        else:
            classified = design["classified"][:, index]
            annotated = design["annotated"][:, index]

        if event == "any":
            statistic = ratio(design, annotated & classified, classified)
            precision = statistic(all_rows)
            low, high = bootstrap(statistic, design, replicates, seed)
            rows.append(metric("precision", event, strictness, precision, low, high,
                               int((annotated & classified).sum()), int(classified.sum()),
                               "bootstrap"))
        else:
            stratum = design["stratum"] == "classified:" + event
            successes = int((annotated & stratum).sum())
            total = int(stratum.sum())
            precision, low, high = wilson(successes, total)
            rows.append(metric("precision", event, strictness, precision, low, high,
                               successes, total, "wilson"))

            # the same articles seen from the other side: how often the classifier picks up a
            # hazard in an article it was given to no class at all
            unclassified = design["stratum"] == "unclassified"
            missed = int((annotated & unclassified).sum())
            rate, miss_low, miss_high = wilson(missed, int(unclassified.sum()))
            rows.append(metric("miss_rate_unclassified", event, strictness, rate,
                               miss_low, miss_high, missed, int(unclassified.sum()), "wilson"))

        statistic = ratio(design, annotated & classified, annotated)
        recall = statistic(all_rows)
        low, high = bootstrap(statistic, design, replicates, seed)
        rows.append(metric("recall", event, strictness, recall, low, high,
                           int((annotated & classified).sum()), int(annotated.sum()), "bootstrap"))

        def f1(rows_, classified=classified, annotated=annotated):
            weight = design["weight"][rows_]
            true_positive = float(weight @ (annotated & classified)[rows_])
            if not true_positive:
                return float("nan")
            precision_ = true_positive / float(weight @ classified[rows_])
            recall_ = true_positive / float(weight @ annotated[rows_])
            return 2 * precision_ * recall_ / (precision_ + recall_)

        low, high = bootstrap(f1, design, replicates, seed)
        rows.append(metric("f1", event, strictness, f1(all_rows), low, high,
                           None, None, "bootstrap"))

        """The same estimates as corpus counts, which is what the panel is built out of"""
        weight = design["weight"]
        rows.append(metric("corpus_true_positives", event, strictness,
                           float(weight @ (annotated & classified)), None, None,
                           None, None, "weighted total"))
        rows.append(metric("corpus_false_positives", event, strictness,
                           float(weight @ (classified & ~annotated)), None, None,
                           None, None, "weighted total"))
        rows.append(metric("corpus_missed", event, strictness,
                           float(weight @ (annotated & ~classified)), None, None,
                           None, None, "weighted total"))
    return rows


def metric(name, hazard, strictness, estimate, low, high, successes, total, method):
    return {"metric": name, "hazard": hazard, "strictness": strictness,
            "estimate": estimate, "ci_low": low, "ci_high": high,
            "k": successes, "n": total, "method": method}


"""Places and countries

The annotation names the place the article reports the hazard as happening in and the country that
place lies in; the key carries the pipeline's counterpart of both, the tagged string the
nearest-keyword rule selected and the country the geocoding maps it to. The comparison is made only
over the classified articles the annotation confirms as reporting a hazard: the place of an article
that reports no hazard is not a thing the pipeline can get right.
"""


def normalise_place(name):
    """Casefolded, unaccented, punctuation-free form of a place name."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", stripped.casefold())).strip()


def same_place(annotated, pipeline):
    """Whether two place names refer to the same place, as far as strings can tell.

    Containment either way accepts the granularity the two sides differ in -- "New Orleans" against
    "New Orleans, Louisiana", "Elbe" against "Elbe-Havel-Land". It cannot bridge the languages: the
    annotation writes an English or a German name and the tagger returns whatever the article used,
    so "Dominican Republic" against "Dominikanische Republik" is scored a mismatch here. The
    country columns, which are ISO3 on both sides, are the language-independent comparison and
    carry the headline result.
    """
    left, right = normalise_place(annotated), normalise_place(pipeline)
    if not left or not right:
        return False
    if left == right:
        return True
    if min(len(left), len(right)) < MIN_CONTAINMENT:
        return False
    return left in right or right in left


def location_frame(sample, design):
    """The classified, annotated-as-reporting articles, with both sides of the comparison."""
    reports_hazard = design["annotated"].any(axis=1)
    frame = sample[(sample["stratum"] != "unclassified") & reports_hazard].copy()

    frame["annotated_place"] = frame["label_location"].fillna("")
    # through the same call gen_coverage_days.py resolves its own countries with, so that the
    # model's ISO3 codes and a human annotator's country names land in the one namespace
    frame["annotated_country"] = frame["label_country"].fillna("").map(to_iso3)
    frame["pipeline_place"] = frame["location_closest"].fillna("")
    frame["pipeline_country"] = frame["location_GID_0"]

    frame["place_exact"] = [
        bool(a) and normalise_place(a) == normalise_place(b)
        for a, b in zip(frame.annotated_place, frame.pipeline_place)]
    frame["place_match"] = [
        same_place(a, b) for a, b in zip(frame.annotated_place, frame.pipeline_place)]
    frame["place_tagged"] = [
        any(same_place(a, tag) for tag in str(tagged).split("|")) if a else False
        for a, tagged in zip(frame.annotated_place, frame.location_tagged.fillna(""))]
    frame["country_match"] = (
        frame.annotated_country.notna() & (frame.annotated_country == frame.pipeline_country))
    return frame


def location_metrics(frame, strictness):
    """Coverage, extraction, mapping and the end-to-end country accuracy."""
    rows = []

    def add(name, successes, total):
        estimate, low, high = wilson(successes, total)
        rows.append(metric(name, "location", strictness, estimate, low, high,
                           successes, total, "wilson"))

    total = len(frame)
    named = frame.annotated_place != ""
    attributed = frame.pipeline_country.notna()

    add("annotation_names_a_place", int(named.sum()), total)
    add("pipeline_attributes_a_country", int(attributed.sum()), total)
    add("pipeline_lost_the_country", int((named & ~attributed).sum()), int(named.sum()))

    """Extraction: did the tagger see the place at all, and did the ranking then pick it"""
    both_named = named & (frame.pipeline_place != "")
    add("place_tagged_anywhere", int((frame.place_tagged & named).sum()), int(named.sum()))
    add("place_selected", int((frame.place_match & both_named).sum()), int(both_named.sum()))
    add("place_selected_exact", int((frame.place_exact & both_named).sum()), int(both_named.sum()))

    """Mapping: given the right place, the right country -- this isolates the geocoding"""
    right_place = frame[frame.place_match & frame.annotated_country.notna()]
    add("country_given_right_place",
        int((right_place.annotated_country == right_place.pipeline_country).sum()),
        len(right_place))

    """End to end: the country the article enters the panel under"""
    comparable = frame[frame.annotated_country.notna() & attributed]
    add("country_agreement",
        int((comparable.annotated_country == comparable.pipeline_country).sum()), len(comparable))
    add("country_agreement_over_all", int(frame.country_match.sum()), total)
    return rows


"""The disagreeing articles, so that the numbers can be read back to the text"""


def error_casebook(sample, design, location):
    """One row per article the annotation and the pipeline disagree about, with the reasons."""
    classified = design["classified"]
    annotated = design["annotated"]

    """Location errors, over the articles the location comparison is defined for.

    A place the two sides name differently is only called wrong where they also disagree on the
    country: string matching cannot tell "Dominikanische Republik" from "Dominican Republic", and
    a mismatch that still lands in the right country is far more often the two sides naming the
    same event at different granularity than the pipeline reading the wrong place. The two are
    separated here so that the casebook can be read without wading through the naming differences;
    the place_selected rate above counts both together and is the conservative figure.
    """
    location_errors = {}
    for row in location.itertuples():
        errors = []
        annotated_country = (None if row.annotated_country is None
                             or pd.isna(row.annotated_country) else row.annotated_country)
        if row.annotated_place and not row.place_match:
            errors.append("place_differs_same_country"
                          if annotated_country and annotated_country == row.pipeline_country
                          else "wrong_place")
        if annotated_country:
            if pd.isna(row.pipeline_country):
                errors.append("no_country")
            elif annotated_country != row.pipeline_country:
                errors.append("wrong_country")
        if errors:
            location_errors[row.sample_id] = errors

    kinds = []
    for index, sample_id in enumerate(sample.sample_id):
        errors = []
        if classified[index].any() and not annotated[index].any():
            errors.append("false_positive")
        if (annotated[index] & ~classified[index]).any():
            errors.append("missed_hazard")
        kinds.append("|".join(errors + location_errors.get(sample_id, [])))

    frame = sample.copy()
    frame["error_type"] = kinds
    frame["annotated_country_iso3"] = frame.sample_id.map(
        dict(zip(location.sample_id, location.annotated_country)))

    columns = ["sample_id", "stratum", "classifier_output", "error_type"]
    columns += ["grade_" + e.lower() for e in EVENTS if "grade_" + e.lower() in frame]
    columns += [c for c in ["label_location", "annotated_country_iso3", "country_name",
                            "location_closest", "location_GID_0", "location_keyword",
                            "location_status", "location_rank", "date", "journal", "title",
                            "notes", "link"] if c in frame]
    return frame[frame.error_type != ""][columns]


"""Report"""


def line(label, row, width=34, successes=False):
    """One metric as a line of the printed report.

    *successes* prints the numerator beside the denominator, which the miss rates need: they rest
    on a handful of articles each, and the count is what tells the reader how much the interval of
    the recall extrapolated from them can carry.
    """
    if row["estimate"] is None or pd.isna(row["estimate"]):
        return "  {:<{}} {:>9}".format(label, width, "--")
    if row["method"] == "weighted total":
        return "  {:<{}} {:>9,.0f}".format(label, width, row["estimate"])
    text = "  {:<{}} {:>8.1%}".format(label, width, row["estimate"])
    if row["ci_low"] is not None and not pd.isna(row["ci_low"]):
        text += "  [{:>5.1%}, {:>5.1%}]".format(row["ci_low"], row["ci_high"])
    if row["n"] and successes:
        text += "  ({:,} of {:,})".format(int(row["k"]), int(row["n"]))
    elif row["n"]:
        text += "  (n = {:,})".format(int(row["n"]))
    return text


def print_report(metrics, location, counts, metadata, population, replicates):
    strictnesses = list(dict.fromkeys(row["strictness"] for row in metrics))
    lookup = {(row["metric"], row["hazard"], row["strictness"]): row for row in metrics}

    print("\nTest sample: {:,} annotated articles, {:,} unclear left out".format(
        counts["n"], counts["unclear"]))
    print("Populations: {} sizes, {:,.0f} articles in the weighted corpus".format(
        population, counts["corpus"]))
    print("Intervals: Wilson within strata, {:,} stratified bootstrap replicates otherwise".format(
        replicates))

    for strictness in strictnesses:
        if strictness in STRICTNESS:
            print("\n\n{} grading: {} counts as reporting on the hazard".format(
                strictness.capitalize(), " or ".join(STRICTNESS[strictness])))
        else:
            print("\n\nAnnotated labels")

        print("\nPrecision of the classifier, within the stratum of each hazard:")
        for event in EVENTS + ["any"]:
            print(line(event, lookup[("precision", event, strictness)]))

        print("\nArticles the classifier assigned to nothing, annotated as reporting on:")
        for event in EVENTS:
            print(line(event, lookup[("miss_rate_unclassified", event, strictness)],
                       successes=True))

        print("\nRecall over the corpus (Horvitz-Thompson), and F1:")
        for event in EVENTS + ["any"]:
            recall = lookup[("recall", event, strictness)]
            f1 = lookup[("f1", event, strictness)]
            print("{}   F1 {:>5.1%}".format(line(event, recall), f1["estimate"]))

        print("\nExtrapolated to the corpus:")
        for event in EVENTS + ["any"]:
            print("  {:<12} {:>11,.0f} true  {:>11,.0f} false  {:>11,.0f} missed".format(
                event,
                lookup[("corpus_true_positives", event, strictness)]["estimate"],
                lookup[("corpus_false_positives", event, strictness)]["estimate"],
                lookup[("corpus_missed", event, strictness)]["estimate"]))

    print("\n\nPlace and country, over the {:,} classified articles annotated as reporting a "
          "hazard".format(location[0]["n"] if location else 0))
    labels = [
        ("annotation_names_a_place", "annotation names a place"),
        ("pipeline_attributes_a_country", "pipeline attributes a country"),
        ("pipeline_lost_the_country", "  ... but not where a place is named"),
        ("place_tagged_anywhere", "annotated place is tagged somewhere"),
        ("place_selected", "  ... and is the one selected"),
        ("place_selected_exact", "  ... by exact string match"),
        ("country_given_right_place", "country right, given the right place"),
        ("country_agreement", "country agrees where both attribute"),
        ("country_agreement_over_all", "country agrees, over all articles"),
    ]
    location_lookup = {row["metric"]: row for row in location}
    for name, label in labels:
        if name in location_lookup:
            print(line(label, location_lookup[name], width=38))


def interval(row):
    """A point estimate and its interval in per cent, as the tables print them."""
    if row["ci_low"] is None or pd.isna(row["ci_low"]):
        return "{:.1f}".format(100 * row["estimate"])
    return "{:.1f} [{:.1f}, {:.1f}]".format(
        100 * row["estimate"], 100 * row["ci_low"], 100 * row["ci_high"])


def hazard_table(metrics, strictness):
    """The tabular of the hazard metrics, under one reading of the annotation grades.

    The manuscript reports the lenient reading, which is the concept the dictionary is built to
    detect: an article carrying a keyword of the impact type. The strict reading stays in
    validation_metrics.csv and in the printed report, where the gap between the two shows how
    much of the classified coverage is a passing mention. Caption and table environment sit in
    the manuscript, as they do for the other tables of the paper.
    """
    lookup = {(row["metric"], row["hazard"], row["strictness"]): row for row in metrics}
    if ("precision", "any", strictness) not in lookup:  # a sheet without grades
        strictness = sorted({row["strictness"] for row in metrics})[0]

    # the manuscript speaks of impact types rather than hazards, and spells the two temperature
    # ones out; the table follows its wording rather than the column names of the annotation
    labels = {"Heat": "Extreme heat", "Cold": "Extreme cold", "any": "Any impact type"}

    lines = [
        "% written by src/evaluate_annotation.py",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Impact type & Precision & Recall & $F_1$ \\\\",
        "\\midrule",
    ]
    for event in EVENTS + ["any"]:
        name = labels.get(event, event)
        lines.append("{} & {} & {} & {} \\\\".format(
            name,
            interval(lookup[("precision", event, strictness)]),
            interval(lookup[("recall", event, strictness)]),
            interval(lookup[("f1", event, strictness)])))
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotation", default=ANNOTATION_AI,
                        help="annotation sheet to evaluate (default: the model annotation)")
    parser.add_argument("--key", default=KEY, help="unblinding key of the sample")
    parser.add_argument("--population", choices=["effective", "raw"], default="effective",
                        help="stratum sizes to extrapolate with (default: effective, the raw "
                             "pool scaled by the share of candidates that survive filtering)")
    parser.add_argument("--strictness", choices=sorted(STRICTNESS), default="lenient",
                        help="grading used for the location comparison and the LaTeX table")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP, metavar="N",
                        help="bootstrap replicates for the weighted estimates")
    parser.add_argument("--no-write", action="store_true",
                        help="print the report without writing any file")
    args = parser.parse_args()

    metadata = load_metadata()
    sample, unclear = load_sample(args.annotation, args.key)
    seed = metadata.get("seed", 0)

    graded = ("grade_" + EVENTS[0].lower()) in sample
    strictnesses = sorted(STRICTNESS) if graded else ["annotated"]

    metrics = []
    designs = {}
    for strictness in strictnesses:
        designs[strictness] = build_design(sample, metadata, args.population, strictness)
        metrics += hazard_metrics(designs[strictness], args.bootstrap, seed, strictness)

    primary = args.strictness if graded else "annotated"
    location = location_frame(sample, designs[primary])
    location_rows = location_metrics(location, primary)

    counts = {"n": len(sample), "unclear": unclear,
              "corpus": float(designs[primary]["weight"].sum())}
    print_report(metrics, location_rows, counts, metadata, args.population, args.bootstrap)

    if args.no_write:
        return

    pd.DataFrame(metrics + location_rows).to_csv(METRICS_FILE, index=False)
    errors = error_casebook(sample, designs[primary], location)
    errors.to_csv(ERRORS_FILE, index=False)
    with open(HAZARD_TABLE_FILE, "w") as stream:
        stream.write(hazard_table(metrics, primary))

    print("\n\nWrote {:,} metrics to {}".format(len(metrics) + len(location_rows), METRICS_FILE))
    print("Wrote {:,} disagreeing articles to {} ({} grading)".format(
        len(errors), ERRORS_FILE, primary))
    print("Wrote the supplementary table to {}".format(HAZARD_TABLE_FILE))


if __name__ == "__main__":
    main()
