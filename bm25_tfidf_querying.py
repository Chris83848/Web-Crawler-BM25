import os
import json
import pickle
import re
import math
from collections import Counter, defaultdict
from nltk.corpus import stopwords

class Indexer:
    def __init__(self, directory="pages"):

        # Store file directory and corpus list
        self.directory = directory
        self.corpus = []

        # Create dictionary mapping to files
        self.did2fn = {}

        # Store vocabulary mappings to term frequencies
        self.tok2idx = {}
        self.idx2tok = {}

        # Store inverted index
        self.postings = {}

        # Store document lengths for BM25 ranking
        self.doc_lengths = []

        # Initialize average document length
        self.avgdl = 0

        # Load and store stopwords
        self.stop_words = set(stopwords.words("english"))

        # Load dataset
        self.load_corpus()

    def create_index(self):
        # Loop through documents in corpus
        for document in self.corpus:
            # Take document id, preprocess its text, and store length
            doc_id = document["doc_id"]
            tokenization = self.text_preprocessing(document["text"])
            self.doc_lengths.append(len(tokenization))

            # Count word frequencies
            term_counts = Counter(tokenization)

            for token, frequency in term_counts.items():
                # Add word to vocabularies if new
                if token not in self.tok2idx:
                    token_id = len(self.tok2idx)
                    self.tok2idx[token] = token_id
                    self.idx2tok[token_id] = token

                # Add to inverted index if new, otherwise update index
                token_id = self.tok2idx[token]
                if token_id not in self.postings:
                    self.postings[token_id] = []
                self.postings[token_id].append((doc_id, frequency))

        # Find average document length
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)

    def load_corpus(self):
        # Loop through files
        for i, filename in enumerate(sorted(os.listdir(self.directory))):
            # Skip over any uncrawled documents
            if not filename.endswith(".json"):
                continue

            # Create proper file path for document
            file_path = os.path.join(self.directory, filename)

            # Load JSON into dictionary
            with open(file_path, "r", encoding="utf-8") as file:
                page_data = json.load(file)

            # Skip bibliography page
            if "bibliography" in page_data["url"]:
                continue

            # Create document object
            document = {
                "doc_id": i,
                "url": page_data["url"],
                "title": page_data["title"],
                "text": page_data["text"]
            }

            # Add document to corpus and store mapping
            self.corpus.append(document)
            self.did2fn[i] = filename

    def text_preprocessing(self, text):
        # Convert text to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r"<.*?>", " ", text)

        # Remove punctuation, numbers, and any other special characters
        text = re.sub(r"[^a-z\s]", " ", text)

        # Eliminate extra spaces in the text
        text = re.sub(r"\s+", " ", text)

        # Create and return a list of the tokenized words, using spaces between them as the delimiter
        tokenization = text.strip().split()

        # Remove stopwords
        tokenization = [token for token in tokenization if token not in self.stop_words]

        return tokenization

    def save_index(self, filename="index.pkl"):
        # Create data dictionary
        data = {
            "avgdl": self.avgdl,
            "tok2idx": self.tok2idx,
            "idx2tok": self.idx2tok,
            "corpus": self.corpus,
            "did2fn": self.did2fn,
            "postings": self.postings,
            "doc_lengths": self.doc_lengths
        }

        # Write data
        with open(filename, "wb") as file:
            pickle.dump(data, file)

    def load_index(self, filename="index.pkl"):
        # Read data
        with open(filename, "rb") as file:
            data = pickle.load(file)

        self.avgdl = data["avgdl"]
        self.tok2idx = data["tok2idx"]
        self.idx2tok = data["idx2tok"]
        self.corpus = data["corpus"]
        self.did2fn = data["did2fn"]
        self.postings = data["postings"]
        self.doc_lengths = data["doc_lengths"]

class SearchAgent:
    def __init__(self, indexer):
        self.indexer = indexer

        # Set BM25 parameters
        self.k1 = 1.5
        self.b = 0.75
        self.delta = 1.0

    def bm25_query(self, query):
        # Retrieve preprocessed text
        tokenization = self.indexer.text_preprocessing(query)
        # Create dictionary for BM25 score
        scores = defaultdict(float)
        # Store document total of corpus
        doc_total = len(self.indexer.corpus)

        # Loop through each word in query
        for token in tokenization:
            # Skip terms not in vocabulary
            if token not in self.indexer.tok2idx:
                continue

            # Find postings list for term using token_id
            token_id = self.indexer.tok2idx[token]
            postings = self.indexer.postings[token_id]

            # Find document and inverse document frequencies
            doc_frequency = len(postings)
            inverse_doc_frequency = math.log((doc_total - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1)

            # Calculate scores for each document
            for doc_id, term_frequency in postings:
                doc_len = self.indexer.doc_lengths[doc_id]
                avgdl = self.indexer.avgdl
                score = inverse_doc_frequency * ((term_frequency * (self.k1 + 1)) / (term_frequency + self.k1 * (1 - self.b + self.b * (doc_len / avgdl)))) + self.delta
                scores[doc_id] += score

        # Rank documents in order and display top five results
        ranked_results = sorted(scores.items(), key=lambda  x: x[1], reverse=True)
        self.display_results(ranked_results[:5])

    def tfidf_query(self, query):
        # Retrieve preprocessed text
        tokenization = self.indexer.text_preprocessing(query)
        # Create dictionary for TF-IDF score
        scores = defaultdict(float)
        # Store document total of corpus
        doc_total = len(self.indexer.corpus)

        # Loop through each word in query
        for token in tokenization:
            # Skip terms not in vocabulary
            if token not in self.indexer.tok2idx:
                continue

            # Find postings list for term using token_id
            token_id = self.indexer.tok2idx[token]
            postings = self.indexer.postings[token_id]

            # Find document and inverse document frequencies
            doc_frequency = len(postings)
            inverse_doc_frequency = math.log(doc_total / (doc_frequency + 1))

            # Calculate scores for each document
            for doc_id, term_frequency in postings:
                scores[doc_id] += term_frequency * inverse_doc_frequency

        # Rank documents in order and display top five results
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.display_results(ranked_results[:5])


    def display_results(self, results):
        # Loop through results, then retrieve and print document information
        for doc_id, score in results:
            doc = self.indexer.corpus[doc_id]
            print("DocID:", doc_id)
            print("Score:", score)
            print("URL:", doc["url"])
            print("filename:", self.indexer.did2fn[doc_id])
            print()


if __name__ == "__main__":
    # Create instances of the Indexer and SearchAgent classes
    indexer = Indexer()
    searchAgent = SearchAgent(indexer)

    # Load/Create index
    if os.path.exists("index.pkl"):
        print("Loading index...")
        indexer.load_index()
    else:
        print("Creating index...")
        indexer.create_index()
        indexer.save_index()

    # Loop through user queries until user quits the program
    while True:
        query = input("Enter query: ")
        if (query.lower() == "quit"):
            break
        query_method = input("bm25 or tfidf?")
        if query_method == "bm25":
            searchAgent.bm25_query(query)
        else:
            searchAgent.tfidf_query(query)