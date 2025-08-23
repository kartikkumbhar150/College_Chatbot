import requests
from bs4 import BeautifulSoup
import re


urls = [
    "https://engg.dypvp.edu.in",
    "https://engg.dypvp.edu.in/college-profile.aspx",
    "https://engg.dypvp.edu.in/computer-engineering.aspx",
    "https://engg.dypvp.edu.in/NIRF-2025.aspx",
    "https://engg.dypvp.edu.in/ugadmissions.aspx",
    "https://engg.dypvp.edu.in/from-the-desk-of-tpo.aspx",
    "https://engg.dypvp.edu.in/training-and-placement-team.aspx",
    "https://engg.dypvp.edu.in/computer-engineering-laboratory.aspx"
]

def clean_text(text):
    text = re.sub(r"\s+", " ", text)  
    return text.strip()

data = []

for url in urls:
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "lxml")

    page_text = soup.get_text(separator=" ", strip=True)
    data.append(clean_text(page_text))



with open("college_data.txt", "w", encoding="utf-8") as f:
    for content in data:
        f.write(content + "\n\n")

print("Website data saved.")
