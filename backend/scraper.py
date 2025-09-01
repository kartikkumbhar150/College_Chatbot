import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
from collections import deque

BASE_URL = "https://engg.dypvp.edu.in"
DOMAIN = "engg.dypvp.edu.in"

visited = set()
data = {}

def clean_text(text):
    text = re.sub(r"\s+", " ", text)  
    return text.strip()

def get_internal_links(url):
    """Extract all internal links from a page"""
    links = set()
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full_url = urljoin(url, href)
            parsed = urlparse(full_url)
            # only keep internal, non-fragment links
            if DOMAIN in parsed.netloc:
                links.add(full_url.split("#")[0])
    except Exception as e:
        print(f"Failed to fetch links from {url}: {e}")
    return links

def scrape_page(url):
    """Scrape text from a single page"""
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "lxml")
        page_text = soup.get_text(separator=" ", strip=True)
        return clean_text(page_text)
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""

def crawl(start_url):
    """Recursive BFS crawler"""
    queue = deque([start_url])
    
    while queue:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        print(f"Crawling: {url}")
        text = scrape_page(url)
        if text:
            data[url] = text

        new_links = get_internal_links(url)
        for link in new_links:
            if link not in visited:
                queue.append(link)

# Start crawling
crawl(BASE_URL)

# Save all scraped content
with open("data/college.txt", "w", encoding="utf-8") as f:
    for url, content in data.items():
        f.write(f"URL: {url}\n")
        f.write(content + "\n\n" + "="*100 + "\n\n")

print(f"Done. Scraped {len(data)} pages. Saved to college.txt")
