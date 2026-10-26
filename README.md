# Coverage Classification

**Which climate disasters does the German press report on — and where in the world did they happen?**

This pipeline processes an 83 GB corpus of German-language news articles and
detects coverage of five climate impacts: **flood, wildfire, heat, storm and
cold**. For each relevant article it identifies the country where the event
took place. The result is a monthly panel of how many days the German press
covered each type of event in each country.

The classifier's accuracy is checked against a human- and AI-annotated test
sample, including a test for geographic bias.

**At a glance**

- End-to-end NLP pipeline: from raw text to an analysis-ready dataset
- Transformer-based named entity recognition on a GPU, run on an HPC cluster (Slurm)
- LLM-assisted annotation (Google Gemini Batch API) combined with human annotation
- Statistical validation: precision, recall, F1 with confidence intervals
- **Stack:** Python, SQLite (FTS5), flair, spaCy, PyTorch, geopy, Gemini API, Slurm

## How it works

```
News corpus ─► Keyword classification ─► Location extraction (NER) ─► Geocoding ─► Coverage-days panel
                                                                                         │
                     Test sample ─► Human + AI annotation ─► Precision / recall / geographic bias
```

| Step | What happens | Tools |
| --- | --- | --- |
| Impact classification | Matches articles against a hand-curated keyword dictionary ([`data/bag_of_words.yml`](data/bag_of_words.yml)) | SQLite full-text search |
| Location extraction | Finds place names in the article text | flair `ner-german-large` (XLM-RoBERTa transformer), GPU |
| Location ranking | Picks the location closest to the impact keyword | spaCy sentence segmentation |
| Geocoding | Maps place names to countries | Nominatim via geopy |
| Aggregation | Builds the monthly coverage-days panel per country and impact type | pandas |

## Validation

Pipeline output is not taken at face value:

1. **Sampling** — a blinded, stratified sample of articles, including extra
   samples per continent.
2. **Annotation** — labelled by a human and by Google Gemini via the Batch
   API, with graded labels
   (*about* / *mention* / *none*) and the main location.
3. **Evaluation** — precision, recall and F1 of the classifier and accuracy of
   the country assignment, with Wilson and bootstrap confidence intervals.
4. **Bias check** — the same metrics split by continent, to test whether the
   pipeline works equally well for all world regions.

Precision is reported in a strict and a lenient version: a keyword match
shows that an article *mentions* an event, but not necessarily that it is
*about* it.

## Repository layout

| Folder | Content |
| --- | --- |
| [`src/`](src/) | One script per pipeline and validation step |
| [`lib/`](lib/) | Shared modules: keyword dictionary, data loading, configuration |
| [`slurm/`](slurm/) | Job chain for running the pipeline on an HPC cluster — see [slurm/README.md](slurm/README.md) |
| [`data/`](data/) | Keyword dictionary; corpus and outputs not included (size and copyright) |

## Running it

```bash
pip install -r requirements.txt          # install torch separately for your CPU/CUDA setup
python -m spacy download de_core_news_sm
python src/check_news_db.py              # checks the corpus at data/news021024.db
```

Then run the scripts in `src/` in pipeline order, or submit the whole chain
to a Slurm cluster (see [slurm/README.md](slurm/README.md)). The news corpus
itself is not included and must be supplied separately.

## Use of AI tools

Parts of the scripts in this repository were written with the help of
the AI coding assistant Claude Code. All
code was reviewed, tested and integrated by me, and the research design,
methods and validation approach are my own.
