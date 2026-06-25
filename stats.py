import os
import json
import re
from collections import Counter, defaultdict
from nltk.corpus import stopwords

# Initialize constant stopwords list
STOPWORDS = set(stopwords.words("english"))

def preprocessing(text):
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
    return tokenization

def load_corpus(directory="pages"):
    # Initialize list to store text of each page
    documents = []

    for filename in os.listdir(directory):
        # Skip over any uncrawled documents
        if not filename.endswith(".json"):
            continue

        # Create proper file path for document
        file_path = os.path.join(directory, filename)

        # Load JSON into dictionary
        with open(file_path, "r", encoding = "utf-8") as file:
            page_data = json.load(file)
        # Skip bibliography page
        if "bibliography" in page_data["url"]:
            continue
        # Add document text only to list
        documents.append(page_data["text"])

    # Return list of documents text
    return documents

def find_statistics(corpus):
    # Initialize collection frequency counter
    collection_frequency = Counter()
    # Initialize document frequency counter
    document_frequency = defaultdict(int)
    # Initialize total words counter
    word_total = 0
    # Initialize word total counter per document
    page_lengths = []

    for document in corpus:
        # Retrieve tokens of document
        tokenization = preprocessing(document)

        # Add length of tokenization to word total of the corpus
        word_total += len(tokenization)

        # Append length of tokenization to word total counter per document
        page_lengths.append(len(tokenization))

        # Update collection frequency using current tokenization
        collection_frequency.update(tokenization)

        # Keep unique tokens from tokenization, remove any further duplicates
        unique_tokens = set(tokenization)

        # Keep count of how many documents contain each unique token
        for token in unique_tokens:
            document_frequency[token] += 1

    # Return all found statistics
    return collection_frequency, document_frequency, word_total, page_lengths

def print_frequent_words(collection_frequency, document_frequency, title):
    print(title)
    # Print 30 most common words in collection_frequency
    for word, frequency in collection_frequency.most_common(30):
        print(f"{word:15} Collection Frequency = {frequency:5} | Document Frequency ={document_frequency[word]:4}")


# Only run code if file is run
if __name__ == "__main__":
    # Retrieve corpus
    corpus = load_corpus()
    # Retrieve corpus statistics
    collection_frequency, document_frequency, word_total, page_lengths = find_statistics(corpus)

    # Print word total of corpus
    print(f"Total Number of Words in Corpus: {word_total}")
    # Print unique word total of corpus
    print(f"Total Number of Unique Words in Corpus: {len(collection_frequency)}")
    # Print average/mean page length up to two decimal points, taken by dividing sum of page_lengths by its length
    print(f"Average Page Length of Corpus: {sum(page_lengths) / len(page_lengths):.2f}")

    # Print frequent words with their collection and document frequencies
    print_frequent_words(collection_frequency, document_frequency, "Top 30 Words in Corpus:")

    # Filter out stopwords from collection_frequency
    filtered_collection_frequency = Counter()
    for word, frequency in collection_frequency.items():
        if word not in STOPWORDS:
            filtered_collection_frequency[word] = frequency

    # Print new frequent words with their collection and document frequencies
    print_frequent_words(filtered_collection_frequency, document_frequency, "Top 30 Non-Stopwords in Corpus:")