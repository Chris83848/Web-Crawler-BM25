# Web Crawler & BM25/TF-IDF Search Engine

**A from-scratch information retrieval pipeline: crawl a website, build an inverted index, and rank search results using BM25 and TF-IDF.**

## What This Is

This project implements three core stages of a real search engine, end to end:

1. **Crawl** — a breadth-first web crawler collects pages from a target domain, extracting and storing each page's title and body text.
2. **Index** — collected pages are tokenized, cleaned, and compiled into an inverted index mapping every term to the documents it appears in.
3. **Rank** — user queries are scored against the indexed corpus using two different ranking models, **BM25** and **TF-IDF**, so their results can be directly compared.

The crawler was run against the [Stanford NLP Information Retrieval textbook](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) as a demonstration corpus.

## How It Works

### 1. Crawling (`crawler.py`)

The crawler starts from a seed URL and explores breadth-first using a queue (`frontier`), staying strictly within the seed's domain. For each page:
- HTML is parsed with BeautifulSoup to extract the page title (`<h1>`) and body text (all `<p>` tags)
- The page is saved as a JSON record (URL, access timestamp, title, text) to the `pages/` directory
- All hyperlinks on the page are discovered, resolved to absolute URLs, and queued only if they're on the same domain, use HTTP/HTTPS, and point to an actual HTML page

A one-second delay between requests keeps the crawler polite to the target server.

### 2. Indexing (`bm25_tfidf_querying.py` — `Indexer` class)

Every crawled page is loaded, preprocessed (lowercased, HTML-stripped, punctuation/numbers removed, tokenized, stopwords removed via NLTK), and compiled into an **inverted index**: a mapping from each unique term to the list of documents it appears in, along with how many times it appears in each (term frequency). Document lengths are tracked alongside the index, since both ranking models below need them.

### 3. Ranking (`bm25_tfidf_querying.py` — `SearchAgent` class)

Both ranking models score documents against a query by summing per-term contributions, but they calculate those contributions differently:

**TF-IDF** scores each document as the sum, across query terms, of *term frequency × inverse document frequency*. This is simple and effective, but with one structural weakness: it has no built-in way to account for document length, so longer documents can score higher purely by virtue of containing more words, and not necessarily because they're more relevant.

**BM25** improves on this with length normalization: it weighs term frequency relative to a document's length compared to the corpus's *average* document length, controlled by tunable parameters `k1` (term frequency saturation) and `b` (how strongly length is normalized). This keeps a long document from automatically outscoring a short, highly relevant one.

## Results

Querying *"computer science department"* against both models on the same corpus produced different top results. BM25 favored documents with strong term concentration regardless of length, while TF-IDF's rankings skewed toward longer documents that simply contained the query terms more times in raw count. This matches the expected theoretical difference between the two models and was the central finding of the original report (full report in [`docs/Report.pdf`](docs/Assignment_Report.pdf)).

## Corpus Statistics

Stats from the demonstration crawl of the Stanford IR textbook site:

| Metric | Value |
|---|---|
| Total documents crawled | 292 |
| Total words in corpus | 816,892 |
| Unique words in corpus | 10,737 |
| Average page length | 2,732.08 words |

**Top 10 most frequent words (excluding stopwords):**

| Word | Collection Frequency | Document Frequency |
|---|---|---|
| documents | 4,535 | 181 |
| document | 4,244 | 174 |
| page | 3,648 | 288 |
| query | 3,313 | 158 |
| figure | 3,059 | 131 |
| may | 2,812 | 287 |
| two | 2,729 | 157 |
| set | 2,554 | 119 |
| term | 2,517 | 144 |
| case | 2,504 | 287 |

*(Full top-30 breakdown available by running `stats.py` against the included corpus.)*

## Repository Structure

```
Web-Crawler-BM25/
├── crawler.py                   # Breadth-first web crawler
├── bm25_tfidf_querying.py       # Indexer + SearchAgent (BM25/TF-IDF ranking)
├── stats.py                     # Corpus statistics (word counts, frequencies)
├── pages/                       # Crawled corpus (JSON page records)
├── docs/
│   └── Assignment_Report.pdf    # Full original report
└── README.md
```

## Running This Yourself

**Requirements:** Python 3, `beautifulsoup4`, `nltk` (with the `stopwords` corpus downloaded)

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

**Crawl a site** (edit the seed URL inside `crawler.py`, then run):
```bash
python crawler.py
```

**Query the existing corpus:**
```bash
python bm25_tfidf_querying.py
```
On first run, this builds and saves an index (`index.pkl`) from the `pages/` directory. This file is gitignored and regenerates automatically, so don't worry if you don't see it in the repo. You'll be prompted to enter a query and choose `bm25` or `tfidf` as the ranking method.

**View corpus statistics:**
```bash
python stats.py
```

## Author

**Christopher Harris** — [GitHub](https://github.com/Chris83848) · [LinkedIn](https://www.linkedin.com/in/christopher-harris9/)
