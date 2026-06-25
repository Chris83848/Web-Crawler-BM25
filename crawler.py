import os
import time
import json
from collections import deque
from urllib.request import urlopen
from urllib.parse import urljoin, urlparse
from datetime import datetime, UTC
from bs4 import BeautifulSoup

class Crawler:
    def __init__(self, seed_url, output_directory = "pages", max_pages = 300):
        # Crawl starting point
        self.seed_url = seed_url

        # Keep crawling contained under same domain
        self.domain = urlparse(seed_url).netloc

        # Initialize frontier queue starting with the seed URL
        self.frontier = deque()
        self.frontier.append(seed_url)

        # Keep track of visited pages
        self.visited = set()

        # Store crawled pages in separate folder and make sure it exists
        self.output_directory = output_directory
        self.max_pages = max_pages
        os.makedirs(self.output_directory, exist_ok=True)

    def extract_info(self, html):
        # Initialize BeautifulSoup HTML
        beautiful_soup = BeautifulSoup(html, "html.parser")

        # Find the title of the web page
        title_label = beautiful_soup.find("h1")

        # Extract title from title label if it exists, otherwise initialize an empty title string
        if title_label:
            title = title_label.get_text(strip = True)
        else:
            title = ""

        # Find all text/paragraph labels
        text_labels = beautiful_soup.find_all("p")

        # Create text string by appending each text line in text labels to each other while removing white space and adding new lines
        text = ""
        for text_lines in text_labels:
            text += text_lines.get_text(strip = True) + "\n"

        # Return the extracted content as a tuple
        return title, text

    def save_page(self, url, title, text, id):
        # Create page record dictionary including URL, time accessed, title, and text
        page_record = {
            "url": url,
            "access_time": datetime.now(UTC).isoformat(),
            "title": title,
            "text": text
        }

        # Creates file name and path using unique id
        file_path = os.path.join(self.output_directory, f"doc_{id}.json")

        # Open file in write mode and close automatically
        with open(file_path, "w", encoding = "utf-8") as file:
            # Put created data dictionary into JSON
            json.dump(page_record, file, ensure_ascii = False, indent = 2)

    def discover_urls(self, html, current_url):
        # Initialize BeautifulSoup HTML
        beautiful_soup = BeautifulSoup(html, "html.parser")

        # Find all hyperlinks
        for link in beautiful_soup.find_all("a", href = True):
            # Turn the found, local URL into a complete, global URL
            complete_url = urljoin(current_url, link["href"])

            # Ensure URL has not been visited before and is valid to follow before appending to the frontier queue to be crawled
            if complete_url not in self.visited and self.is_url_ok_to_follow(complete_url):
                self.frontier.append(complete_url)

    def is_url_ok_to_follow(self, url):
        # Break URL into parts to check
        parsed_url = urlparse(url)

        # Only crawl web pages
        if parsed_url.scheme not in ("http", "https"):
            return False
        # Only crawl within same domain to prevent wandering outside main website
        if parsed_url.netloc != self.domain:
            return False
        # Only crawl HTML pages
        if not parsed_url.path.endswith((".html", ".htm")):
            return False

        # All checks passed, return true
        return True

    def crawl(self):
        # Initialize id number counter for documents
        id = 0

        # Crawl if there is an available URL to crawl in the frontier queue and the max page limit has not been hit yet
        while self.frontier and id < self.max_pages:
            # Take the next available URL in queue by following breadth-first search
            current_url = self.frontier.popleft()

            # Skip URL if it has already been crawled
            if current_url in self.visited:
                continue

            # Attempt to retrieve the web page
            try:
                opened_url = urlopen(current_url)

                # Only accept successful response code
                if opened_url.getcode() != 200:
                    continue

                # Decode HTML to a string
                html = opened_url.read().decode("utf-8", errors ="ignore")

            # Skip the URL if there's an error
            except Exception:
                continue

            # Add crawled URL to visited list
            self.visited.add(current_url)

            # Take the title and text of the page
            title, text = self.extract_info(html)

            # Save the page and its data
            self.save_page(current_url, title, text, id)

            # Find new page links to crawl
            self.discover_urls(html, current_url)

            # Update id counter
            id += 1

            # Pause crawling for one second
            time.sleep(1)

# Initialize and run crawler
crawler = Crawler("https://nlp.stanford.edu/IR-book/information-retrieval-book.html")
crawler.crawl()