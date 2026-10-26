"""
Tag the locations in every article of one impact type with flair's German NER model
(ner-german-large). Stage 2 of the pipeline; runs best on a GPU.

    python src/classify_locations.py <cold|flood|heat|storm|wildfire>
"""

from flair.data import Sentence
from flair.nn import Classifier
from flair.models import SequenceTagger
from flair.splitter import SegtokSentenceSplitter
import pandas as pd
from tqdm import tqdm
import sys

dtype = sys.argv[1]


data = pd.read_parquet("data/eventcoverage/eventcoverage_"+dtype+".parquet")

splitter = SegtokSentenceSplitter()
tagger = SequenceTagger.load("flair/ner-german-large")

data["flair_locations"] = None

for i, article in tqdm(data.iterrows(), total=len(data)):

    sentences = splitter.split(article["text"])

    tagger.predict(sentences)

    locations = [entity.text for sentence in sentences for entity in sentence.get_spans('ner') if entity.tag == 'LOC']

    data.at[i, "flair_locations"] = locations

data.to_parquet("data/eventcoverage/eventcoverage_"+dtype+"_localised.parquet")
