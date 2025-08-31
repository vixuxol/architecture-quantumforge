import requests
import os
from bs4 import BeautifulSoup

# список тем был сформирован вручную
PAGES_TO_DOWNLOAD = [
    "Alex_Louis_Armstrong", # персонаж
    "Alphonse_Elric", # персонаж
    "Alchemy", # описание термина
    "Amestris", # страна
    "Blood_Rune", # описание термина
    "Creta", # страна
    "Edward_Elric", # персонаж
    "Envy", # персонаж
    "Father", # персонаж
    "Greed", # персонаж
    "Homunculus", # описание термина
    "Ishval", # страна
    "Laboratory_3", # локация
    "Laboratory_5", # локация
    "Lust", # персонаж
    "Maes_Hughes", # персонаж
    "King_Bradley", # персонаж
    "Philosopher's_Stone", # описание термина
    "Pinako Rockbell", # персонаж
    "Religion", # описание термина
    "Riza_Hawkeye", # персонаж
    "Roy_Mustang", # персонаж
    "Selim_Bradley", # персонаж
    "Scar", # персонаж
    "The_Gate", # описание термина
    "The_Truth", # описание термина
    "Trisha_Elric", # персонаж
    "Van_Hohenheim", # персонаж
    "Winry_Rockbell", # персонаж
    "Xerxes", # страна

]

KNOWLEDGE_BASE_PATH = "../knowledge_base/before_processing_base"

def extract_text_from_url(url, output_file="extracted_text.txt"):
    """
    Извлекает текст из всех <p> тегов на странице и сохраняет в файл
    """
    output_file_path = os.path.join(KNOWLEDGE_BASE_PATH, output_file)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        paragraphs = soup.find_all('p')
        
        text_content = []
        for p in paragraphs:
            text = p.get_text().strip()
            if text:
                text_content.append(text)
        
        # Сохраняем в файл
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_content))
        
        print(f"Успешно извлечено {len(text_content)} параграфов")
        print(f"Сохранено в файл: {output_file}")
    except Exception as e:
        print(f"Ошибка: {e}")
    
if __name__ == "__main__":
    WIKI_DOMAIN = "https://fma.fandom.com"
    WIKI_BASE_URL = "/wiki/"
    for topic in PAGES_TO_DOWNLOAD:
        extract_text_from_url(WIKI_DOMAIN + WIKI_BASE_URL + topic, f"{topic}.txt")