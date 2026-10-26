"""
Attribute every classified article to a country and aggregate the result into the
coverage-days panel: per country, impact type and month, the number of days the German
press reported on it. Stage 6 of the pipeline.
"""

import pandas as pd
import numpy as np
import countrynames

# Configuration
DISASTER_TYPES = {
    "Heat wave": "heat",
    "Cold wave": "cold",
    "Flood": "flood",
    "Storm": "storm",
    "Wildfire": "wildfire",
}
INFILE = "data/eventcoverage/eventcoverage_{dtype}_localised_ranked.parquet"
COLUMNS = [
    "date", "journal", "link", "flair_locations",
    "keyword_distance_words", "keyword_distance_chars",
    "same_sentence", "location_rank",
]
START_DATE = pd.to_datetime("1990-01-01")
END_DATE = pd.to_datetime("2023-12-31")

# The shortest location name the corpus reports on has three letters, so a
# shorter string is never a place: it is an initial or a fragment left by the
# text extraction -- "L" from the HTML entity in "L&uuml;neburg", "B" from
# "Badezimmer", "St." from an abbreviated saint -- and nominatim resolves those
# to whole voivodeships, départements and São Tomé. Punctuation is not part of
# a name and is not counted, see location_length()
MIN_LOCATION_LENGTH = 3

# Nominatim geocoding of map_location_to_country.py
GEOCODED = "data/localisation/all_locations_geocoded.pkl"


def load_geocoding(path=GEOCODED):
    """The geocoding the tagged locations are mapped through, one row per location.

    Indexed by the lower cased location, and restricted to what an article can
    be attributed by: Nominatim features of the type "administrative" that carry
    a country code. The country name and the feature type are kept for
    generate_test_sample.py, which records what a string was mapped to; the
    attribution itself needs only the location -> country code dictionary
    returned by location_mapping().
    """
    geocoding = pd.read_pickle(path)[["country", "country_code", "location_type"]]
    geocoding = geocoding[geocoding.country_code.notna()]
    return geocoding[geocoding.location_type == "administrative"]


def location_mapping(geocoding):
    """Lower cased location -> Nominatim's two letter country code."""
    return geocoding.country_code.to_dict()


def location_length(name):
    """The length of a tagged string counted in letters and digits alone.

    The text extraction leaves punctuation on its fragments -- "St.", "D.C",
    "B&" -- so the characters that make a name are counted rather than the
    characters of the string.
    """
    return sum(1 for character in name if character.isalnum())


def map_location(location, mapping):
    """The ISO3 country code of a single location, None where it names none.

    A location has no country code when it is too short to be a place name,
    when it did not geocode at all, or when it geocoded to a feature that lies
    in no single country, e.g. an ocean or a continent.

    The geocoding is used as Nominatim returns it, without manual corrections:
    a hand-verified correction table would only cover the European countries it
    was audited for, which would bias the measure rather than correct it.
    """
    if location is None or (not isinstance(location, str) and pd.isna(location)):
        return None
    name = str(location).lower()
    if location_length(name) < MIN_LOCATION_LENGTH:
        return None
    country_code = mapping.get(name)
    if country_code is None:
        return None
    country = countrynames.to_code_3(country_code)
    return country


def select_location(locations, ranks, mapping):
    """The highest ranked location of an article that names a country.

    The one rule by which an article is placed, shared with
    generate_test_sample.py so that the validation grades exactly the choice the
    panel is built on. ranking_locations.py aligns the ranks with the locations
    flair tagged: the locations sharing a sentence with a keyword of the impact
    type come first, the rest follow by their distance to the nearest keyword,
    ties are broken by the character distance and the order of appearance, and a
    location it could not place in the text is left unranked (NaN).

    Locations that name no country -- oceans, continents, strings the geocoder
    did not resolve or too short to be a place, see map_location() -- are
    skipped rather than attributed to nothing, so an article naming "Europa"
    next to the keyword and Spain a few words later is attributed to Spain.

    Returns (index into *locations*, ISO3 code, position in the ranking counted
    from 1, number of ranked locations); the first three are None where no
    ranked location names a country.
    """
    if locations is None or not hasattr(locations, "__len__"):
        return None, None, None, 0
    ranks = [] if ranks is None else ranks
    candidates = sorted(
        (ranks[i], i)
        for i in range(len(locations))
        if i < len(ranks) and not pd.isna(ranks[i])
    )
    for position, (_, i) in enumerate(candidates, start=1):
        country = map_location(locations[i], mapping)
        if country is not None:
            return i, country, position, len(candidates)
    return None, None, None, len(candidates)


def located_country(locations, ranks, distances_words, distances_chars, same_sentence,
                    mapping):
    """The country of the highest ranked location of the article.

    An article reports on an event somewhere, so it is attributed to the single
    place it names next to the impact keyword rather than to every place it
    mentions, the one select_location() picks.

    Returns the ISO3 code along with the sentence flag and the distances it was
    selected at, which record how firmly the article is tied to the place it is
    attributed to. An article carrying the keywords of several impact types is
    placed once per type and kept once per type, so the distances no longer
    decide between those attributions.
    """
    i, country, _, _ = select_location(locations, ranks, mapping)
    if i is None:
        return None, False, np.nan, np.nan
    return country, bool(same_sentence[i]), distances_words[i], distances_chars[i]


def main():
    mapping = location_mapping(load_geocoding())

    # Process each disaster type
    articles = None

    for dtype, name in DISASTER_TYPES.items():
        # Load data, ranked by the distance of every location to the nearest keyword
        news_data = pd.read_parquet(INFILE.format(dtype=name), columns=COLUMNS)

        # Filter rows
        news_data = news_data.loc[news_data.journal != "zeit_offline"]
        news_data = news_data.loc[(news_data.date >= START_DATE) & (news_data.date <= END_DATE)]

        # Attribute every article to the country of its highest ranked location
        located = [
            located_country(
                row.flair_locations, row.location_rank,
                row.keyword_distance_words, row.keyword_distance_chars,
                row.same_sentence, mapping,
            )
            for row in news_data.itertuples()
        ]
        news_data["GID_0"] = [l[0] for l in located]
        news_data["same_sentence"] = [l[1] for l in located]
        news_data["distance_words"] = [l[2] for l in located]
        news_data["distance_chars"] = [l[3] for l in located]
        news_data["dtype"] = dtype
        news_data = news_data.dropna(subset=["GID_0"])
        print(f"{dtype}: {len(news_data):,} of {len(located):,} articles attributed to a country")

        # Concatenate with previous data
        columns = ["date", "journal", "link", "GID_0", "same_sentence", "distance_words", "distance_chars", "dtype"]
        articles = pd.concat([articles, news_data[columns]]) if articles is not None else news_data[columns]

    # Aggregate by date, journal, country and impact type
    coverage_days = articles.groupby(["date", "journal", "GID_0", "dtype"]).size().to_frame("articles").reset_index()
    monthly_coverage_days = coverage_days.set_index("date").groupby([pd.Grouper(freq="ME"), "journal", "GID_0", "dtype"]).articles.count().to_frame("coverage_days")
    monthly_coverage_days.to_csv("data/coverage_days_per_dtype.csv")

    # Whether an impact type was covered at all, country-month by country-month.
    # The coverage days above are counted per journal, so a country-month counts as
    # covered for an impact type when any journal reported on that type there in
    # that month. The dummies are laid out on the full grid of countries the corpus
    # reports on and months it is active in, so zeros are explicit: a country-month the
    # aggregation holds no row for is a genuine zero, i.e. the journals published
    # and none of them reported on that impact type in that country.
    dtype_dummies = (
        monthly_coverage_days.reset_index()
        .assign(covered=1)
        .pivot_table(index=["date", "GID_0"], columns="dtype", values="covered",
                     aggfunc="max", fill_value=0)
    )
    dtype_dummies = dtype_dummies.reindex(columns=list(DISASTER_TYPES), fill_value=0)
    dtype_dummies.columns = [f"coverage_{DISASTER_TYPES[impact]}" for impact in dtype_dummies.columns]
    grid = pd.MultiIndex.from_product(
        [
            monthly_coverage_days.index.get_level_values("date").unique().sort_values(),
            pd.Index(sorted(monthly_coverage_days.index.get_level_values("GID_0").unique()), name="GID_0"),
        ],
        names=["date", "GID_0"],
    )
    dtype_dummies = dtype_dummies.reindex(grid, fill_value=0).astype(int)
    dtype_dummies.to_csv("data/coverage_dtype_dummies.csv")
    del dtype_dummies, grid

    del monthly_coverage_days, coverage_days

    # Aggregate across disaster types. An article carrying the keywords of several
    # impact types is attributed to each of them separately
    joint_coverage_days = articles.groupby(["date", "journal", "GID_0"]).size().to_frame("articles").reset_index()
    del articles

    # Bin into article count intervals
    bins = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 30)])
    joint_coverage_days["bins"] = pd.cut(joint_coverage_days["articles"], bins)

    # Count article days by month and bin
    monthly_bin_counts = (
        joint_coverage_days
        .groupby([pd.Grouper(key="date", freq="ME"), "bins", "journal", "GID_0"]).articles
        .size()
        .to_frame("coverage_days")
        .reset_index()
    )
    monthly_bin_counts.to_csv("data/coverage_days_monthly_bins.csv", index=False)

    # Aggregate by month
    joint_coverage_days = joint_coverage_days.set_index("date").groupby([pd.Grouper(freq="ME"), "journal", "GID_0"]).articles.count().to_frame("coverage_days")
    joint_coverage_days.to_csv("data/coverage_days.csv")


if __name__ == "__main__":
    main()
