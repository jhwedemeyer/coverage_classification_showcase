"""
The keyword dictionary of the impact classification, and the full text search it turns into.

news_fts is an FTS5 table on the default unicode61 tokenizer, which neither stems nor splits
compounds: a query for "Waldbrand" matches the token waldbrand and nothing else, so an article
that only ever writes "Waldbränden" is not found. A keyword ending in * is therefore expanded
into an FTS5 prefix query, which matches every token beginning with it.
"""

import yaml

PATH = 'data/bag_of_words.yml'


def load(path=PATH):
    with open(path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def stems(keywords):
    """The keywords without their prefix star, for matching outside the full text search.

    ranking_locations.py looks the keywords up in the article text with a substring regex, which
    already reaches the inflections and compounds the star stands for in the search, so there the
    star is nothing but a character that matches no article.
    """
    return [keyword.strip().rstrip("*").strip() for keyword in keywords]


def fts_query(keywords):
    """The FTS5 MATCH expression matching an article that carries any one of *keywords*."""
    terms = []
    for keyword in keywords:
        keyword = keyword.strip()
        if keyword.endswith("*"):
            terms.append('"{}"*'.format(keyword[:-1].strip()))
        else:
            terms.append('"{}"'.format(keyword))
    return " OR ".join(terms)
