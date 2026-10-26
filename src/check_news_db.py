"""
Check that the news database matches what the pipeline expects, and rebuild its full text
index if it is missing.

A missing database and a missing FTS5 index both show up in stage 1 as "no such table:
news_fts"; this script tells the two apart.

    python src/check_news_db.py                 # report on the database
    python src/check_news_db.py --build-fts     # (re)build the index (takes hours)
"""

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from config import NEWS_DB_PATH as DEFAULT_DB

# The schema the local database was built with; the triggers keep the index in
# step with later inserts, deletes and updates of `news`.
FTS_DDL = """
CREATE VIRTUAL TABLE news_fts USING fts5(
    title,
    text,
    date UNINDEXED,
    journal UNINDEXED,
    link UNINDEXED,
    content='news',
    content_rowid='id'
)
"""

TRIGGER_DDL = {
    "news_ai": """
        CREATE TRIGGER news_ai AFTER INSERT ON news
        BEGIN
            INSERT INTO news_fts (rowid, title, text)
            VALUES (new.id, new.title, new.text);
        END
    """,
    "news_ad": """
        CREATE TRIGGER news_ad AFTER DELETE ON news
        BEGIN
            INSERT INTO news_fts (news_fts, rowid, title, text)
            VALUES ('delete', old.id, old.title, old.text);
        END
    """,
    "news_au": """
        CREATE TRIGGER news_au AFTER UPDATE ON news
        BEGIN
            INSERT INTO news_fts (news_fts, rowid, title, text)
            VALUES ('delete', old.id, old.title, old.text);
            INSERT INTO news_fts (rowid, title, text)
            VALUES (new.id, new.title, new.text);
        END
    """,
}


def connect(path, read_only=True):
    """Open the database without creating it if it is not there."""
    if not os.path.exists(path):
        return None
    uri = f"file:{os.path.abspath(path)}" + ("?mode=ro" if read_only else "")
    return sqlite3.connect(uri, uri=True)


def objects(con):
    """The tables, virtual tables and triggers in the database.

    Read from sqlite_master directly: an sqlite without the FTS5 module still
    lists `news_fts` here, it just cannot query it, and telling that apart from
    an index that is genuinely absent is the point of this script.
    """
    return {
        name: kind
        for kind, name in con.execute("SELECT type, name FROM sqlite_master")
    }


def has_fts5(con):
    """Whether this sqlite build knows the FTS5 module at all."""
    try:
        con.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(x)")
        con.execute("DROP TABLE temp.fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def report(path, full=False):
    """Describe the database, and return whether stage 1 can run against it.

    Every check is one that answers in a moment: over a corpus of this size a
    `count(*)` reads the whole table and takes many minutes, which is what
    `--full` is for.
    """
    print(f"database   : {path}")
    print(f"sqlite     : {sqlite3.sqlite_version} (python {sys.version.split()[0]})")

    if not os.path.exists(path):
        print("             DOES NOT EXIST")
        print()
        print("The corpus is not in the repository (*.db is git-ignored), so a")
        print("fresh clone has no database and sqlite creates an empty one on")
        print("first connect -- which is what 'no such table' then means. Copy")
        print("it over, e.g.")
        print(f"    rsync -avP <source>:{path} {path}")
        return False

    size = os.path.getsize(path)
    print(f"size       : {size / 1e9:.1f} GB")

    con = connect(path)
    fts5 = has_fts5(con)
    print(f"fts5 module: {'available' if fts5 else 'NOT COMPILED INTO THIS SQLITE'}")

    present = objects(con)
    if size < 100_000 or not present:
        print("             EMPTY -- the file sqlite creates for a path that")
        print("             does not exist, not the corpus. Copy the real one.")
        return False

    if "news" in present:
        # max(id) reads the primary key index instead of the table; the exact
        # count is only asked for with --full.
        if full:
            articles = con.execute("SELECT count(*) FROM news").fetchone()[0]
            print(f"news       : {articles:,} articles")
        else:
            articles = con.execute("SELECT max(id) FROM news").fetchone()[0]
            print(f"news       : up to id {articles:,} (exact count with --full)")
    else:
        print("news       : MISSING -- nothing to search and nothing to index")
        return False

    if "news_fts" not in present:
        print("news_fts   : NOT IN THE SCHEMA -- the index stage 1 searches")
        print("             this database has the articles but no full text index;")
        print("             build it over `news` with")
        print("                 python src/check_news_db.py --build-fts")
        return False

    print("news_fts   : in the schema")

    if not fts5:
        print()
        print("The index is there, but this sqlite cannot read it: the module it")
        print("is built with is not compiled in. That is a matter of the python")
        print("the pipeline runs under, not of the database -- conda-forge's")
        print("sqlite has FTS5, a bare system python may not. Run the pipeline in")
        print("the environment slurm/setup_env.sh creates.")
        return False

    # exactly what stage 1 does, over a keyword from the flood bag of words
    try:
        hit = con.execute(
            "SELECT title FROM news_fts WHERE news_fts MATCH 'Hochwasser' LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError as e:
        print(f"test query : FAILED -- {e}")
        return False

    if hit is None:
        print("test query : no article matches 'Hochwasser' -- the index is empty")
        print("             rebuild it with --build-fts")
        return False
    print(f"test query : ok, e.g. {hit[0][:60]!r}")

    if full:
        indexed = con.execute("SELECT count(*) FROM news_fts").fetchone()[0]
        print(f"indexed    : {indexed:,} rows")

    missing_triggers = [t for t in TRIGGER_DDL if t not in present]
    if missing_triggers:
        print(f"triggers   : missing {', '.join(missing_triggers)} -- the index")
        print("             will not follow later changes to `news`")

    con.close()
    return True


def build_fts(path):
    """Create the index and fill it from `news`."""
    con = connect(path, read_only=False)
    if con is None:
        sys.exit(f"{path} does not exist")

    present = objects(con)
    if "news" not in present:
        sys.exit(f"{path} has no `news` table to index")
    if not has_fts5(con):
        sys.exit("this sqlite build has no FTS5 module")

    if "news_fts" not in present:
        print("creating news_fts")
        con.execute(FTS_DDL)

    for name, ddl in TRIGGER_DDL.items():
        if name not in present:
            print(f"creating trigger {name}")
            con.execute(ddl)

    articles = con.execute("SELECT count(*) FROM news").fetchone()[0]
    print(f"indexing {articles:,} articles -- this reads the whole table, expect hours")
    con.execute("INSERT INTO news_fts (news_fts) VALUES ('rebuild')")
    con.commit()

    indexed = con.execute("SELECT count(*) FROM news_fts").fetchone()[0]
    print(f"done: {indexed:,} rows indexed")
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", nargs="?", default=DEFAULT_DB)
    parser.add_argument(
        "--build-fts",
        action="store_true",
        help="create and fill news_fts over the `news` table",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="count the rows exactly, which reads the whole table (minutes)",
    )
    args = parser.parse_args()

    if args.build_fts:
        build_fts(args.database)

    sys.exit(0 if report(args.database, args.full) else 1)
