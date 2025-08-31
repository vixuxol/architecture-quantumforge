import json
import os
import re
from pathlib import Path

def load_replacement_dict(file_path):
    """
    Загружает словарь замен из JSON файла
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Файл {file_path} не найден!")
        return {}
    except json.JSONDecodeError:
        print(f"Ошибка чтения JSON файла {file_path}!")
        return {}

def replace_with_dict(text, replacement_dict):
    """
    Заменяет слова в тексте согласно словарю
    """
    if not replacement_dict:
        return text
    sorted_keys = sorted(replacement_dict.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        replacement = replacement_dict[key]
        pattern = r'\b' + re.escape(key) + r'\b'
        text = re.sub(pattern, replacement, text)
    
    return text

def process_files(input_folder, output_folder, replacement_dict):
    """
    Обрабатывает все файлы в папке
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    supported_extensions = {'.txt', '.md', '.html', '.xml', '.json', '.csv'}
    
    processed_count = 0
    error_count = 0
    
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = replace_with_dict(content, replacement_dict)
                
                output_file = output_path / file_path.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                processed_count += 1
                print(f"✓ Обработан: {file_path.name}")
                
            except Exception as e:
                error_count += 1
                print(f"✗ Ошибка при обработке {file_path.name}: {e}")
    
    print(f"\nОбработка завершена!")
    print(f"Успешно обработано: {processed_count} файлов")
    print(f"С ошибками: {error_count} файлов")

def main():
    TERMS_MAP_FILE = "../knowledge_base/terms_map.json"
    INPUT_FOLDER = "../knowledge_base/before_processing_base"
    OUTPUT_FOLDER = "../knowledge_base/"
    
    replacement_dict = load_replacement_dict(TERMS_MAP_FILE)
    
    if not replacement_dict:
        print("Словарь замен пуст или не загружен!")
        return
    
    print(f"Загружено {len(replacement_dict)} замен")
    process_files(INPUT_FOLDER, OUTPUT_FOLDER, replacement_dict)

if __name__ == "__main__":
    main()