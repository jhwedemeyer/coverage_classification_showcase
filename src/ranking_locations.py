"""
Rank the locations found by flair by how close they are to an impact keyword in the article.

For each location, the distance to the nearest keyword is measured in words and characters.
Locations in the same sentence as a keyword are ranked first, the rest by distance. The top
ranked location is taken as the place the event happened. Results are stored as arrays aligned
with `flair_locations`.

Keywords are searched in the article body; the headline is used only if the body contains none.
"""

import bisect
import os
import re
import sys

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import bag_of_words as bow

DTYPES = {
    "cold": "Cold",
    "flood": "Flood",
    "heat": "Heat",
    "storm": "Storm",
    "wildfire": "Wildfire",
}

INFILE = "data/eventcoverage/eventcoverage_{dtype}_localised.parquet"
OUTFILE = "data/eventcoverage/eventcoverage_{dtype}_localised_ranked.parquet"

TOKEN = re.compile(r"\w+", re.UNICODE)
SEPARATOR = "\n\n"  # between headline and body, counts as no word

MODEL = "de_core_news_sm"
BATCH_SIZE = 64


def sentence_splitter():
    """spaCy's German sentence segmenter.

    `senter` is the statistical segmenter shipped with the model. It is disabled
    by default in favour of the dependency parser, which segments a little better
    but takes about two thirds longer over the corpus; everything else the model
    brings -- tagger, parser, lemmatizer, NER -- is excluded, none of it is read
    here.
    """
    nlp = spacy.load(
        MODEL,
        exclude=["tagger", "morphologizer", "parser", "lemmatizer",
                 "attribute_ruler", "ner"],
    )
    nlp.enable_pipe("senter")
    nlp.max_length = max(nlp.max_length, 5_000_000)  # a few articles are long
    return nlp


def sentence_offsets(doc, body_start):
    """Start of every sentence of the document, in characters.

    spaCy segments the body. The headline is one sentence of its own, however it
    is punctuated, and is never merged into the first sentence of the body.
    """
    starts = [0] if body_start else []
    return starts + [body_start + sent.start_char for sent in doc.sents]


def sentence_index(sentence_starts, position):
    """Index of the sentence the character position belongs to."""
    return max(bisect.bisect_right(sentence_starts, position) - 1, 0)


def keyword_pattern(keywords):
    """Case insensitive pattern matching the keywords anywhere inside a word.

    Matching substrings (instead of whole words) keeps inflected forms and
    German compounds, e.g. "Hitzewelle" also matches "Hitzewellen" and
    "Sommerhitzewelle". That is what the prefix star of the bag of words asks the
    full text search for, so the star itself is stripped here rather than matched.
    """
    keywords = sorted(bow.stems(keywords), key=len, reverse=True)
    return re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)


def locate(text, locations, body_start=0):
    """Character spans of the locations, in order of appearance in the text.

    flair returns the locations in the order they occur, so each location is
    searched from the end of the previous match onwards. This keeps repeated
    mentions of the same location apart. Nothing before `body_start` is searched:
    flair only ever saw the body, so a location that also appears in the prepended
    headline still belongs to its mention in the body.
    """
    spans = []
    cursor = body_start
    for location in locations:
        start = text.find(location, cursor)
        if start == -1:  # fall back to the first occurrence in the body
            start = text.find(location, body_start)
        if start == -1:
            spans.append(None)
            continue
        spans.append((start, start + len(location)))
        cursor = start + len(location)
    return spans


def span_distance(a, b):
    """Gap between two spans, 0 if they overlap."""
    if b[0] >= a[1]:
        return b[0] - a[1]
    if a[0] >= b[1]:
        return a[0] - b[1]
    return 0


def word_index(token_starts, position):
    """Index of the word the character position belongs to."""
    return max(bisect.bisect_right(token_starts, position) - 1, 0)


def article_distances(text, locations, pattern, body_start, sentence_starts):
    """Distance of every location to the nearest keyword.

    `text` is the headline and the body joined, `body_start` the index at which
    the body begins and `sentence_starts` the sentence boundaries of the two,
    as returned by `sentence_offsets`. Returns (word distances, character
    distances, nearest keywords, same sentence), each aligned with `locations`.
    Entries are NaN/None/False where no distance could be estimated (no keyword
    in the article or location not found in the body).
    """
    n = len(locations)
    words = [np.nan] * n
    chars = [np.nan] * n
    nearest = [None] * n
    same_sentence = [False] * n

    matches = [m.span() for m in pattern.finditer(text)]

    # The headline is the exception, not a part of the text: it is read only
    # where the body itself names no keyword and every distance would otherwise
    # stay NaN. Wherever the body carries the keyword, the distances are the
    # ones of the body alone and a headline mention changes nothing.
    matches = [m for m in matches if m[0] >= body_start] or matches

    if not matches:
        return words, chars, nearest, same_sentence

    match_starts = [m[0] for m in matches]
    token_spans = [m.span() for m in TOKEN.finditer(text)]
    token_starts = [s[0] for s in token_spans]

    keyword_sentences = {sentence_index(sentence_starts, m[0]) for m in matches}

    for i, span in enumerate(locate(text, locations, body_start)):
        if span is None:
            continue

        same_sentence[i] = sentence_index(sentence_starts, span[0]) in keyword_sentences

        # the nearest keyword is among the matches surrounding the location
        pos = bisect.bisect_left(match_starts, span[0])
        candidates = matches[max(pos - 2, 0):pos + 2]
        match = min(candidates, key=lambda m: span_distance(span, m))

        # floats throughout, so that the NaNs do not clash with the integer
        # distances when the arrays are written to parquet
        chars[i] = float(span_distance(span, match))
        words[i] = float(
            abs(word_index(token_starts, span[0]) - word_index(token_starts, match[0]))
        )
        nearest[i] = text[match[0]:match[1]]

    return words, chars, nearest, same_sentence


def rank(words, chars, same_sentence):
    """Rank of every location, 0 for the one the article is about.

    Locations sharing a sentence with a keyword come first, and only among
    themselves does the distance decide. Where no location shares a sentence
    with a keyword the ranking falls back to the raw distance over all of them.
    Ties are broken by the character distance and then by the order of
    appearance, so the ranking does not depend on the order of the arrays.
    Locations without a distance are left unranked (NaN).
    """
    order = sorted(
        (not same_sentence[i], words[i], chars[i], i)
        for i in range(len(words))
        if not np.isnan(words[i])
    )
    ranks = [np.nan] * len(words)
    for position, (_, _, _, i) in enumerate(order):
        ranks[i] = float(position)
    return ranks


def as_list(locations):
    """The locations of one article as a list of strings.

    parquet hands the column back as a numpy array rather than a list, and
    articles without locations as None or NaN.
    """
    if locations is None or not hasattr(locations, "__len__"):
        return []
    return list(locations)


def document(article):
    """Headline and body of one article as one string, and where the body starts."""
    title = article["title"]
    if not isinstance(title, str) or not title.strip():
        return article["text"], 0
    return title + SEPARATOR + article["text"], len(title) + len(SEPARATOR)


def bodies(data):
    """The article bodies, in the order of the rows, missing ones as empty."""
    for text in data["text"]:
        yield text if isinstance(text, str) else ""


def main(dtypes):
    bag_of_words = bow.load()
    nlp = sentence_splitter()

    for dtype in dtypes:
        data = pd.read_parquet(INFILE.format(dtype=dtype))
        pattern = keyword_pattern(bag_of_words[DTYPES[dtype]])

        # the bodies go through spaCy in batches, which is where the segmentation
        # spends its time, and are consumed row by row alongside the articles
        docs = nlp.pipe(bodies(data), batch_size=BATCH_SIZE)

        distances = []
        for (_, article), doc in tqdm(
            zip(data.iterrows(), docs), total=len(data), desc=dtype
        ):
            locations = as_list(article["flair_locations"])
            if not locations or not isinstance(article["text"], str):
                distances.append(([], [], [], [], []))
                continue
            text, body_start = document(article)
            words, chars, nearest, same = article_distances(
                text, locations, pattern, body_start,
                sentence_offsets(doc, body_start),
            )
            distances.append((words, chars, nearest, same, rank(words, chars, same)))

        data["keyword_distance_words"] = [d[0] for d in distances]
        data["keyword_distance_chars"] = [d[1] for d in distances]
        data["nearest_keyword"] = [d[2] for d in distances]
        data["same_sentence"] = [d[3] for d in distances]
        data["location_rank"] = [d[4] for d in distances]

        n_locations = sum(len(d[0]) for d in distances)
        n_missing = sum(np.isnan(v) for d in distances for v in d[0])
        n_same = sum(sum(d[3]) for d in distances)
        n_ranked = sum(any(d[3]) for d in distances if len(d[0]))
        n_articles = sum(1 for d in distances if len(d[0]))
        print(
            f"{dtype}: {len(data):,} articles, {n_locations:,} locations, "
            f"{n_missing:,} without distance "
            f"({n_missing / max(n_locations, 1):.1%}), "
            f"{n_same:,} in a keyword sentence "
            f"({n_same / max(n_locations, 1):.1%}), "
            f"{n_ranked:,} of {n_articles:,} located articles ranked by sentence "
            f"({n_ranked / max(n_articles, 1):.1%})"
        )

        data.to_parquet(OUTFILE.format(dtype=dtype))


if __name__ == "__main__":
    main(sys.argv[1:] or list(DTYPES))
