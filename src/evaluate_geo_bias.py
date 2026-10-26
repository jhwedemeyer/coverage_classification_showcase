"""
Check whether the location attribution or the impact classification works worse for some
continents than for others.

Splits the validation results by continent and prints precision, recall and F1 as CSV:

    python src/evaluate_geo_bias.py                          # country attribution
    python src/evaluate_geo_bias.py --table classification   # impact classification
    python src/evaluate_geo_bias.py --strictness strict

Precision is grouped by the continent the pipeline assigned, recall by the annotated one.
Results are computed on the main sample and on the continent sample (250 classified articles per
continent), which gives under-represented regions such as Africa, Oceania and South America
enough articles for meaningful estimates. Recall cannot be measured on the continent sample,
since missed articles have no assigned continent to draw by. Weighting and intervals follow
evaluate_annotation.py.

Run from the repository root.
"""

import argparse
import contextlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import evaluate_annotation as ev
from ai_annotation import EVENTS

COUNTRIES = "../geo_data/countries-and-meta.pkl"
CONTINENT_ANNOTATION = ev.TESTSAMPLE + "test_sample_continent_annotation_ai.csv"
CONTINENT_KEY = ev.TESTSAMPLE + "test_sample_continent_key.csv"


def continent_lookup():
    """ISO3 -> two-letter continent code, from the table the panel's own Continent column is from."""
    countries = pd.read_pickle(COUNTRIES)
    return dict(zip(countries["ISO3"], countries["Continent"]))


def share(numerator, denominator):
    n = int(denominator.sum())
    return (numerator & denominator).sum() / n if n else float("nan")


def metrics(frame, continent=None):
    """Precision, recall and F1 of one continent, or of all with None."""
    if continent is None:
        attributed = frame.pipeline_continent.notna()
        annotated = frame.continent.notna()
    else:
        attributed = frame.pipeline_continent == continent
        annotated = frame.continent == continent
    right = frame.country_match

    precision = share(right, attributed)
    recall = share(right, annotated)
    bottom = attributed.sum() + annotated.sum()
    f1 = 2 * (right & attributed).sum() / bottom if bottom else float("nan")
    return {"continent": continent or "all", "precision": precision, "recall": recall, "f1": f1,
            "n_attributed": int(attributed.sum()), "n_annotated": int(annotated.sum())}


def with_continents(frame, lookup):
    """*frame* with the annotated and the pipeline's continent, from its two ISO3 columns."""
    frame = frame.copy()
    frame["continent"] = frame.annotated_country.map(lookup)
    frame["pipeline_continent"] = frame.pipeline_country.map(lookup)
    return frame


def continent_table(location):
    """The measures per continent and over all countries, one row each.

    *location* is evaluate_annotation.location_frame(); it is not modified. Reads the country table
    by a path relative to the repository root, the directory the pipeline runs from.
    """
    frame = with_continents(location, continent_lookup())
    continents = frame.continent.value_counts().index.tolist()
    return pd.DataFrame([metrics(frame, c) for c in continents] + [metrics(frame)])


def classification_metrics(frame, design, hazard, continent, replicates, seed):
    """Precision, recall and F1 of the classification of one hazard ("any" for all five together)
    on one continent, or on all with None, each with its interval."""
    if hazard == "any":
        classified = design["classified"].any(axis=1)
        annotated = design["annotated"].any(axis=1)
    else:
        index = EVENTS.index(hazard)
        classified = design["classified"][:, index]
        annotated = design["annotated"][:, index]
    if continent is None:
        on_attributed = frame.pipeline_continent.notna().to_numpy()
        on_annotated = frame.continent.notna().to_numpy()
    else:
        on_attributed = (frame.pipeline_continent == continent).to_numpy()
        on_annotated = (frame.continent == continent).to_numpy()
    attributed = classified & on_attributed
    reporting = annotated & on_annotated
    all_rows = np.arange(len(frame))

    if hazard == "any":
        statistic = ev.ratio(design, annotated & attributed, attributed)
        precision = statistic(all_rows)
        precision_low, precision_high = ev.bootstrap(statistic, design, replicates, seed)
        n_classified = int(attributed.sum())
    else:
        stratum = (design["stratum"] == "classified:" + hazard) & on_attributed
        n_classified = int(stratum.sum())
        precision, precision_low, precision_high = ev.wilson(
            int((annotated & stratum).sum()), n_classified)

    statistic = ev.ratio(design, reporting & classified, reporting)
    recall = statistic(all_rows)
    recall_low, recall_high = ev.bootstrap(statistic, design, replicates, seed)

    def f1(rows):
        precision_ = ev.ratio(design, annotated & attributed, attributed)(rows)
        recall_ = ev.ratio(design, reporting & classified, reporting)(rows)
        if not precision_ + recall_ > 0:
            return float("nan")
        return 2 * precision_ * recall_ / (precision_ + recall_)

    f1_low, f1_high = ev.bootstrap(f1, design, replicates, seed)
    return {"hazard": hazard, "continent": continent or "all",
            "precision": precision, "precision_low": precision_low,
            "precision_high": precision_high,
            "recall": recall, "recall_low": recall_low, "recall_high": recall_high,
            "f1": f1(all_rows), "f1_low": f1_low, "f1_high": f1_high,
            "n_classified": n_classified, "n_annotated": int(reporting.sum()),
            "n_missed": int((reporting & ~classified).sum())}


def classification_table(sample, design, replicates=ev.BOOTSTRAP, seed=0):
    """The classification measures per hazard, per continent and over all countries.

    *sample* and *design* are evaluate_annotation.load_sample() and build_design() of it, over the
    whole sample, the unclassified stratum included. n_classified is the classified articles the
    precision rests on, n_annotated and n_missed the annotated articles and the misses among them
    the recall rests on -- sampled articles, not the weighted estimates.
    """
    frame = with_continents(pd.DataFrame({
        # as evaluate_annotation.location_frame() resolves it
        "annotated_country": sample["label_country"].fillna("").map(ev.to_iso3),
        "pipeline_country": sample["location_GID_0"],
    }), continent_lookup())
    continents = frame.continent.value_counts().index.tolist() + [None]
    return pd.DataFrame([
        classification_metrics(frame, design, hazard, continent, replicates, seed)
        for hazard in EVENTS + ["any"] for continent in continents])


"""The continent sample"""


def load_continent_sample(annotation, key, strictness):
    """The continent sample and its design, or None when it is missing or not annotated yet.

    The design holds what evaluate_annotation.build_design() holds, with the weight of every
    article that of its continent stratum. The key's own `continent` column, the pipeline's
    continent, is dropped: with_continents() derives it again from the pipeline's country.
    """
    if not os.path.exists(annotation):
        print("{} does not exist, leaving the continent sample out".format(annotation))
        return None
    try:
        sample, _ = ev.load_sample(annotation, key)
    except SystemExit as reason:
        print("{}\nLeaving the continent sample out".format(reason))
        return None
    sample = sample.drop(columns="continent")
    strata = ev.load_metadata()["continent_sample"]["strata"]
    rates = {s["stratum"]: s["n_drawn"] / s["population_effective"] for s in strata}
    graded = ("grade_" + EVENTS[0].lower()) in sample
    design = {
        "weight": 1.0 / sample["stratum"].map(rates).to_numpy(),
        "classified": ev.classifier_labels(sample),
        "annotated": ev.annotated_labels(sample, strictness if graded else "annotated"),
        "stratum_code": pd.Categorical(sample["stratum"]).codes,
        "stratum": sample["stratum"].to_numpy(),
    }
    return sample, design


def attribution_frame(sample, design, source):
    """Every classified article of *sample* with what the attribution measures need, one row each.

    The countries are resolved as evaluate_annotation.location_frame() resolves them, but over all
    classified articles rather than the ones the annotation reads as reporting a hazard, which are
    flagged in `reporting`: the bootstrap has to resample whole strata, in which the number of
    reporting articles is itself random. The strata are prefixed with *source*, so that those of
    the two samples stay apart.
    """
    classified = (sample["stratum"] != "unclassified").to_numpy()
    frame = with_continents(pd.DataFrame({
        "annotated_country": sample["label_country"].fillna("").map(ev.to_iso3),
        "pipeline_country": sample["location_GID_0"],
    }), continent_lookup())
    frame["country_match"] = (frame.annotated_country.notna()
                              & (frame.annotated_country == frame.pipeline_country))
    frame["reporting"] = design["annotated"].any(axis=1)
    frame["weight"] = design["weight"]
    frame["stratum"] = source + ":" + sample["stratum"]
    return frame[classified]


def combined_attribution_metrics(frame, design, continent, replicates, seed):
    """Precision, recall and F1 of the attribution of one continent, or of all with None, over
    a weighted frame of attribution_frame() rows, each with its bootstrap interval."""
    if continent is None:
        attributed = frame.pipeline_continent.notna().to_numpy()
        annotated = frame.continent.notna().to_numpy()
    else:
        attributed = (frame.pipeline_continent == continent).to_numpy()
        annotated = (frame.continent == continent).to_numpy()
    reporting = frame.reporting.to_numpy()
    attributed, annotated = attributed & reporting, annotated & reporting
    right = frame.country_match.to_numpy()
    weight = design["weight"]

    def f1(rows):
        bottom = float(weight[rows] @ attributed[rows] + weight[rows] @ annotated[rows])
        return 2 * float(weight[rows] @ (right & attributed)[rows]) / bottom if bottom \
            else float("nan")

    row = {"continent": continent or "all"}
    all_rows = np.arange(len(frame))
    for name, statistic in [("precision", ev.ratio(design, right & attributed, attributed)),
                            ("recall", ev.ratio(design, right & annotated, annotated)),
                            ("f1", f1)]:
        row[name] = statistic(all_rows)
        row[name + "_low"], row[name + "_high"] = ev.bootstrap(statistic, design, replicates, seed)
    row["n_attributed"] = int(attributed.sum())
    row["n_annotated"] = int(annotated.sum())
    return row


def continent_sample_table(sample, design, main_sample, main_design, replicates, seed):
    """The attribution per continent and over all of them, on the continent sample, with the
    classified articles it does not cover taken from the main sample.

    The continent sample covers the classified articles the pipeline attributes to a country of
    one of its continents. The rest of the classified articles -- attributed to no country, or to
    one of another continent -- enter recall's denominator all the same, as articles the
    pipeline lost, and are taken from the classified strata of the main sample. The two frames do
    not overlap, so each article is weighted by the sample that covers its part of the corpus, and
    the pooled estimate is a sum of the two.
    """
    covered = ev.load_metadata()["continent_sample"]["continents"]
    rest = attribution_frame(main_sample, main_design, "main")
    rest = rest[~rest.pipeline_continent.isin(covered)]
    frame = pd.concat([attribution_frame(sample, design, "continent"), rest], ignore_index=True)
    continents = [c for c in frame[frame.reporting].pipeline_continent.value_counts().index
                  if c in covered]
    return weighted_attribution_table(frame, continents, replicates, seed)


def weighted_continent_table(sample, design, replicates=ev.BOOTSTRAP, seed=0):
    """continent_table() on the main sample, Horvitz-Thompson weighted, with bootstrap intervals.

    continent_table() takes plain shares of the sampled articles, which over-represents the
    small classified strata -- a Heat article stands for some 23 corpus articles, a Flood article
    for some 116. Weighted, the measures refer to the classified corpus, as those of the
    continent sample do.
    """
    frame = attribution_frame(sample, design, "main").reset_index(drop=True)
    continents = frame[frame.reporting].continent.value_counts().index.tolist()
    return weighted_attribution_table(frame, continents, replicates, seed)


def weighted_attribution_table(frame, continents, replicates, seed):
    """combined_attribution_metrics() per continent and over all of them, one row each."""
    combined = {"weight": frame.weight.to_numpy(),
                "stratum_code": pd.Categorical(frame.stratum).codes}
    return pd.DataFrame([combined_attribution_metrics(frame, combined, c, replicates, seed)
                         for c in continents + [None]])


def continent_sample_classification(sample, design, replicates, seed):
    """The classification precision per hazard and continent, and over all continents, on the
    continent sample. n_classified is the sampled articles it rests on; recall is left empty."""
    continent = sample["location_GID_0"].map(continent_lookup())
    all_rows = np.arange(len(sample))
    rows = []
    for hazard in EVENTS + ["any"]:
        if hazard == "any":
            classified = design["classified"].any(axis=1)
            annotated = design["annotated"].any(axis=1)
        else:
            index = EVENTS.index(hazard)
            classified = design["classified"][:, index]
            annotated = design["annotated"][:, index]
        for group in continent.value_counts().index.tolist() + [None]:
            if group is None:
                on_continent = continent.notna().to_numpy()
                attributed = classified & on_continent
                statistic = ev.ratio(design, annotated & attributed, attributed)
                precision = statistic(all_rows)
                low, high = ev.bootstrap(statistic, design, replicates, seed)
            else:
                attributed = classified & (continent == group).to_numpy()
                precision, low, high = ev.wilson(int((annotated & attributed).sum()),
                                                 int(attributed.sum()))
            rows.append({"hazard": hazard, "continent": group or "all",
                         "precision": precision, "precision_low": low, "precision_high": high,
                         "n_classified": int(attributed.sum())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotation", default=ev.ANNOTATION_AI,
                        help="annotation sheet to evaluate (default: the model annotation)")
    parser.add_argument("--key", default=ev.KEY, help="unblinding key of the sample")
    parser.add_argument("--strictness", choices=sorted(ev.STRICTNESS), default="lenient",
                        help="grading of the hazard labels (default: lenient)")
    parser.add_argument("--table", choices=["attribution", "classification"],
                        default="attribution",
                        help="country attribution (default) or hazard classification per continent")
    parser.add_argument("--continent-annotation", default=CONTINENT_ANNOTATION,
                        help="annotation sheet of the continent sample (default: the model "
                             "annotation)")
    parser.add_argument("--continent-key", default=CONTINENT_KEY,
                        help="unblinding key of the continent sample")
    parser.add_argument("--bootstrap", type=int, default=ev.BOOTSTRAP, metavar="N",
                        help="bootstrap replicates for the weighted estimates")
    args = parser.parse_args()

    # the loader's notes go to stderr, so that stdout is nothing but the CSV
    with contextlib.redirect_stdout(sys.stderr):
        metadata = ev.load_metadata()
        sample, _ = ev.load_sample(args.annotation, args.key)
        continent_sample = load_continent_sample(
            args.continent_annotation, args.continent_key, args.strictness)
    graded = ("grade_" + EVENTS[0].lower()) in sample
    strictness = args.strictness if graded else "annotated"
    design = ev.build_design(sample, metadata, "effective", strictness)
    seed = metadata.get("seed", 0)

    if args.table == "classification":
        tables = [classification_table(sample, design, args.bootstrap, seed)]
        if continent_sample is not None:
            tables.append(continent_sample_classification(*continent_sample, args.bootstrap, seed))
    else:
        tables = [continent_table(ev.location_frame(sample, design)),
                  weighted_continent_table(sample, design, args.bootstrap, seed)]
        if continent_sample is not None:
            tables.append(continent_sample_table(*continent_sample, sample, design,
                                                 args.bootstrap, seed))
    names = ["main", "continent"] if args.table == "classification" \
        else ["main", "main_weighted", "continent"]
    table = pd.concat([t.assign(sample=name) for t, name in zip(tables, names)],
                      ignore_index=True)
    table = table[["sample"] + [c for c in table.columns if c != "sample"]]
    print(table.to_csv(index=False, float_format="%.3f"), end="")


if __name__ == "__main__":
    main()
