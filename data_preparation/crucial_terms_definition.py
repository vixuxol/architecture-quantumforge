import requests
from faker import Faker
import re
import urllib
import json
from bs4 import BeautifulSoup

STOP_WORDS = ["anime", "series", "chapter", "comic", "theater", "episode", "volume", "movie", "fandom", "fullmetal alchemist", "list", "game", "part", "season", "books", "community", "soundtrack", "manga", "live-action", "/quotes", "/gallery", "/overview", "/history", "ova"]

fake = Faker()

def is_english_alphanumeric(text):
    """
    Проверяет, что строка содержит только английские буквы и цифры
    """
    if not text:
        return False
    
    # Паттерн: только a-z, A-Z и 0-9 и пробелы
    pattern = r'^[a-zA-Z0-9 ]+$'
    return bool(re.match(pattern, text))

def parse_wiki_page(url:str) -> tuple[list[str], str | None]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        list_items = soup.find_all('li')
        
        link_texts = []
        for li in list_items:
            links = li.find_all('a')
            for link in links:
                if link.text.strip():
                    link_texts.append(link.text.strip())
        
        next_page_text = None
        next_page_links = soup.find_all('a', string=re.compile(r'Next Page', re.IGNORECASE))
        
        if next_page_links:
            next_page_text_match = re.search(r'Next page\s*\(([^)]+)\)', next_page_links[0].text, re.IGNORECASE)
            if next_page_text_match:
                next_page_text = next_page_text_match.group(1)
        
        return link_texts, next_page_text
        
    except requests.RequestException as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return [], None
    except Exception as e:
        print(f"Ошибка при парсинге: {e}")
        return [], None

if __name__ == "__main__":
    wiki_domain = "https://fma.fandom.com"
    wiki_path = "/wiki/Special:AllPages"
    wiki_query_params = {"from": "2003+Anime"}

    query = urllib.parse.urlencode(wiki_query_params)

    crucial_terms_raw = set()
    crucial_terms_mapping = {}


    while query:
        link_texts, next_page = parse_wiki_page(wiki_domain + wiki_path + "?" + query)
        crucial_terms_raw.update(link_texts)

        if next_page:
            wiki_query_params = {"from": next_page}
            query = urllib.parse.urlencode(wiki_query_params)
        else:
            query = None

    for crucial_term_raw in crucial_terms_raw:
        if not any(stop_word in crucial_term_raw.lower() for stop_word in STOP_WORDS) and crucial_term_raw not in crucial_terms_mapping and is_english_alphanumeric(crucial_term_raw):
            crucial_terms_mapping[crucial_term_raw] = fake.name()

    with open("./terms_map_raw.json", "w+") as file:
        json.dump(crucial_terms_mapping, file)

