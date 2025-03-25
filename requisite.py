from urllib.parse import urljoin
import time
import json
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from collections import defaultdict
import re
import nltk

BASE_URL = "https://pureportal.coventry.ac.uk/"
DATA_FILE = "publications.json"
INDEX_FILE = "inverted_index.json"


nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = nltk.WordNetLemmatizer()


def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text) 
    text = text.lower()
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def setup_selenium():
    options = Options()
    options.add_argument("--headless=new") 
    options.add_argument("--disable-blink-features=AutomationControlled") 
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")  # Spoof user-agent
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver


def check_robots_txt(base_url, crawl_delay=2):
    driver = setup_selenium()
    driver.get(urljoin(base_url, "/robots.txt"))
    time.sleep(10)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    robots_txt_content = soup.get_text()
    for line in robots_txt_content.splitlines():
        line = line.strip()
        if line.lower().startswith("crawl-delay:"):
            crawl_delay = int(line[len("crawl-delay:"):].strip())
    print(f"Crawl delay set to: {crawl_delay} seconds")
    return crawl_delay

def crawl_publications(base_url, data_file=DATA_FILE):
    print("Starting crawl...")
    crawl_delay = check_robots_txt(base_url)
    print(f"Crawl delay set to: {crawl_delay} seconds")

    driver = setup_selenium()
    print("Selenium WebDriver initialized.")

    publications = []
    author_info = []
    page = 0
    page_limit = 2

    print(f"Crawling author pages (up to page {page_limit})...")
    while page < page_limit:
        print(f"Fetching page {page + 1} of author listings...")
        driver.get(f"{base_url}/en/organisations/fbl-school-of-economics-finance-and-accounting/persons/?page={page}")
        time.sleep(10)
        print("Page loaded. Parsing author details...")

        soup = BeautifulSoup(driver.page_source, "html.parser")
        for author in soup.find_all("h3", class_="title"):
            author_name = author.get_text(strip=True)
            author_link = author.find("a", class_="link person")["href"]
            author_info.append({"name": author_name, "link": urljoin(base_url, author_link)})
            print(f"Found author: {author_name}")

        page += 1

    print(f"Total authors found: {len(author_info)}")
    print("Starting to crawl publication pages for each author...")

    for author in author_info:
        author_url = author['link'] + "/publications"
        print(f"Crawling publications for author: {author['name']} at {author_url}")
        driver.get(author_url)
        time.sleep(10)
        print("Publication page loaded. Parsing publications...")

        soup = BeautifulSoup(driver.page_source, "html.parser")
        for pub_item in soup.find_all("li", class_="list-result-item"):
            pub_title = pub_item.find("h3", class_="title").get_text(strip=True)
            print(f"Found publication: {pub_title}")

            authors = []
            author_details = pub_item.find_all("a", class_="link person")
            for author in author_details:
                author_name = author.get_text(strip=True)
                author_profile = author['href']
                authors.append({'name': author_name, 'link': author_profile})
                print(f"Found author of publication: {author_name}")

            pub_year = pub_item.find("div", class_="search-result-group").get_text(strip=True) if pub_item.find("div", class_="search-result-group") else None
            if pub_year is None:
                print("No publication year found. Skipping this publication.")
                continue

            journal_info = pub_item.find("span", class_="journal")
            journal_name = journal_info.get_text(strip=True) if journal_info else None
            volume_info = pub_item.find("span", class_="volume")
            volume = volume_info.get_text(strip=True) if volume_info else None
            pub_link = pub_item.find("a", class_="link")["href"]

            publication_data = {
                "title": pub_title,
                "authors": authors,
                "publication_year": pub_year,
                "journal": journal_name,
                "volume": volume,
                "link": urljoin(base_url, pub_link),
            }
            publications.append(publication_data)
            print(f"Added publication: {pub_title}")

    print(f"Total publications crawled: {len(publications)}")
    print(f"Saving publications to {data_file}...")

    with open(data_file, "w") as f:
        json.dump(publications, f, indent=4)

    print(f"Data saved to {data_file}.")
    return publications

def load_publications(data_file=DATA_FILE):
    try:
        with open(data_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("No existing data found. Please run the crawler first.")
        return []

# Indexer functions
def build_index(publications, index_file=INDEX_FILE):
    inverted_index = defaultdict(list)
    for doc_id, pub in enumerate(publications):
        title_tokens = preprocess_text(pub["title"])
        for token in title_tokens:
            inverted_index[token].append(doc_id)
    with open(index_file, "w") as f:
        json.dump(inverted_index, f, indent=4)
    print(f"Inverted index built and saved to {index_file}.")
    return inverted_index

def load_index(index_file=INDEX_FILE):
    try:
        with open(index_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("No existing index found. Please build the index first.")
        return {}

# Main function to run crawler and indexer
def main():
    publications = crawl_publications(BASE_URL)
    build_index(publications)

if __name__ == "__main__":
    main()