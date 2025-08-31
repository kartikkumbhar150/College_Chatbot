import requests
from bs4 import BeautifulSoup
import re
import os

# ✅ List of links you want to scrape (manually provided)
URLS = [
    "https://engg.dypvp.edu.in",
    "https://engg.dypvp.edu.in/computer-engineering.aspx",
    "https://engg.dypvp.edu.in/ugadmissions.aspx",
    "https://engg.dypvp.edu.in/college-profile.aspx",
    "https://engg.dypvp.edu.in/vision-mission.aspx",
    "https://engg.dypvp.edu.in/major-highlights-of-TP-cell.aspx",
    "https://engg.dypvp.edu.in/training-and-placement-team.aspx",
    "https://engg.dypvp.edu.in/Infrastructure.aspx"

    # add more links here
]

SAVE_PATH = "data/college.txt"
os.makedirs("data", exist_ok=True)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)  
    return text.strip()

def scrape_page(url):
    """Scrape text from a single page"""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        page_text = soup.get_text(separator=" ", strip=True)
        return clean_text(page_text)
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return ""

# ✅ Scrape all given URLs
scraped_data = {}
for url in URLS:
    print(f"🌐 Scraping: {url}")
    text = scrape_page(url)
    if text:
        scraped_data[url] = text

# ✅ Save all scraped content
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    for url, content in scraped_data.items():
        f.write(f"URL: {url}\n")
        f.write(content + "\n\n" + "="*100 + "\n\n")

print(f"✅ Done. Scraped {len(scraped_data)} pages. Saved to {SAVE_PATH}")
