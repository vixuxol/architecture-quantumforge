import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import time
import json
from pathlib import Path

class ChromaDBSearcher:
    def __init__(self, 
                 persist_directory: str = "./vector_index",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: str = "knowledge_base"):
        """
        Инициализация поиска по ChromaDB
        
        Параметры:
        - persist_directory: папка для хранения базы данных
        - embedding_model_name: название модели для эмбеддингов
        - collection_name: название коллекции
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self._get_or_create_collection()
        
        print(f"ChromaDB инициализирована")
        print(f"Модель: {embedding_model_name}")
        print(f"Размер эмбеддингов: {self.embedding_size}")
        print(f"Коллекция: {collection_name}")
    
    def _get_or_create_collection(self):
        """Получает или создает коллекцию"""
        try:
            collection = self.client.get_collection(self.collection_name)
            print(f"Загружена существующая коллекция")
            return collection
        except ValueError:
            print(f"Создана новая коллекция: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
            )
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               where: Dict = None,
               where_document: Dict = None) -> List[Dict]:
        """
        Поиск по запросу в ChromaDB
        
        Параметры:
        - query: текстовый запрос
        - n_results: количество результатов
        - where: фильтр по метаданным
        - where_document: фильтр по содержимому документов
        
        Возвращает:
        - Список результатов с документами, метаданными и расстояниями
        """
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Выполнение поиска
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )
        
        # Форматирование результатов
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'score': 1 - results['distances'][0][i]  # Преобразование расстояния в score
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict:
        """Получает информацию о коллекции"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_dimension": self.embedding_size,
            "persist_directory": self.persist_directory
        }


def interactive_search():
    """Интерактивный режим поиска"""
    
    chroma_searcher = ChromaDBSearcher()
    
    print("🔍 Интерактивный поиск по ChromaDB")
    print("Введите запрос (или 'quit' для выхода):")
    
    while True:
        try:
            query = input("\nПоисковый запрос: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Выполняем поиск
            start_time = time.time()
            results = chroma_searcher.search(query, n_results=5)
            search_time = time.time() - start_time
            
            print(f"\nНайдено результатов: {len(results)} (время: {search_time:.3f}с)")
            print("=" * 80)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. 📄 {result['metadata'].get('source', 'Unknown')}")
                print(f"   ⭐ Score: {result['score']:.3f}")
                print(f"   📝 {result['document'][:200]}...")
                print(f"   📊 Metadata: {result['metadata']}")
                print()
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка поиска: {e}")

if __name__ == "__main__": 
    # Запуск интерактивного поиска
    interactive_search()