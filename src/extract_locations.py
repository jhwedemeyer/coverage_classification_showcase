"""
Collect the unique location strings tagged across all five impact types, as input for
geocoding. Stage 3 of the pipeline.
"""

import pandas as pd
import itertools

locations = []
flair_locations = []

for dtype in ["cold", "flood", "heat", "storm", "wildfire"]:
    data = pd.read_parquet(
        f"data/eventcoverage/eventcoverage_{dtype}_localised.parquet",
        columns=["flair_locations"],
    )
    flair_locations += itertools.chain.from_iterable(data.flair_locations.dropna().values)

flair_locations = list(set(flair_locations))
flair_locations = pd.DataFrame(flair_locations, columns=["location"])

flair_locations.to_pickle("data/localisation/all_locations.pkl")
