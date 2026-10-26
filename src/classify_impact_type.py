"""
Find the articles reporting on each of the five impact types with a keyword search over the
news corpus (SQLite FTS5). Stage 1 of the pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import bag_of_words as bow
from news_data_loader import NewsData
from config import NEWS_DB_PATH

"""Load bag of words"""

bag_of_words = bow.load()

"""Generate Coverage Timeseries"""
nd = NewsData(NEWS_DB_PATH)

dirname = "data/eventcoverage/"

if not os.path.exists(dirname):
    os.makedirs(dirname)

for event in ["Flood", "Wildfire", "Heat", "Storm", "Cold"]:
    EVENT_BAG_OF_WORDS = bow.fts_query(bag_of_words[event])
    coverage = nd.search(EVENT_BAG_OF_WORDS,columns = ["title", "text", "date", "link", "journal"])
    # parquet rather than pickle, so the files are portable across pandas versions
    coverage.to_parquet(dirname + "eventcoverage_"+event.lower().replace(" ","_")+".parquet")
