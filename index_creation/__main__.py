import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import chromadb
from chromadb.config import Settings

class VectorIndexBuilder:
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_db_type: str = "chroma"):
        """
        Инициализация строителя векторного индекса
        
        Параметры:
        - embedding_model_name: название модели для эмбеддингов
        - vector_db_type: тип векторной БД ('faiss' или 'chroma')
        представлена реалищация только для chromadb, но можно потом расширить скрипт
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_db_type = vector_db_type
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        
        print(f"Модель: {embedding_model_name}")
        print(f"Размер эмбеддингов: {self.embedding_size}")
        print(f"Векторная БД: {vector_db_type}")
    
    def load_documents(self, folder_path: str) -> List[Document]:
        """Загружает документы из папки"""
        documents = []
        folder = Path(folder_path)
        
        supported_extensions = {'.txt', '.md', '.html', '.pdf'}
        exclude_path = folder / "before_processing_base"
        
        for file_path in folder.rglob('*'):
            if exclude_path in file_path.parents:
                continue
            if file_path.suffix.lower() in supported_extensions and file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                            "filepath": str(file_path),
                            "filetype": file_path.suffix
                        }
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    print(f"Ошибка чтения файла {file_path}: {e}")
        
        print(f"Загружено {len(documents)} документов")
        return documents
    
    def split_documents(self, documents: List[Document], 
                       chunk_size: int = 500, 
                       chunk_overlap: int = 50) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Создано {len(chunks)} чанков")
        return chunks
    
    def generate_embeddings(self, chunks: List[Document]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"Генерация эмбеддингов для {len(texts)} чанков...")
        
        start_time = time.time()
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        end_time = time.time()
        
        print(f"Эмбеддинги сгенерированы за {end_time - start_time:.2f} секунд")
        return embeddings
    
    def build_faiss_index(self, chunks: List[Document], embeddings: np.ndarray, output_path: str):
        raise NotImplementedError("FAISS представлен как возможность расширения скрипта")
    
    def build_chroma_index(self, chunks: List[Document], embeddings: np.ndarray, output_path: str):
        """Строит Chroma индекс"""
        client = chromadb.PersistentClient(path=output_path)
        
        collection = client.create_collection("knowledge_base")
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk.page_content for chunk in chunks]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            metadatas.append({
                "source": chunk.metadata.get("source", ""),
                "filename": chunk.metadata.get("filename", ""),
                "chunk_index": i
            })
        
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Chroma индекс сохранен в {output_path}")
        return collection
    
    def build_index(self, input_folder: str, output_path: str, 
                   chunk_size: int = 500, chunk_overlap: int = 50):
        print("Начало построения векторного индекса...")
        start_time = time.time()
        
        # 1. Загрузка документов
        documents = self.load_documents(input_folder)
        if not documents:
            print("Не найдено документов для обработки")
            return
        
        # 2. Разбиение на чанки
        chunks = self.split_documents(documents, chunk_size, chunk_overlap)
        
        # 3. Генерация эмбеддингов
        embeddings = self.generate_embeddings(chunks)
        
        # 4. Построение индекса
        if self.vector_db_type == "faiss":
            index, metadata = self.build_faiss_index(chunks, embeddings, output_path)
        elif self.vector_db_type == "chroma":
            index = self.build_chroma_index(chunks, embeddings, output_path)
        else:
            raise ValueError(f"Неизвестный тип векторной БД: {self.vector_db_type}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nПостроение индекса завершено!")
        print(f"Общее время: {total_time:.2f} секунд")
        print(f"Количество чанков: {len(chunks)}")
        print(f"Размер эмбеддингов: {self.embedding_size}")
        
        # Сохраняем информацию о построении
        self.save_build_info(input_folder, output_path, len(chunks), total_time)
        
        return index
    
    def save_build_info(self, input_folder: str, output_path: str, 
                       num_chunks: int, total_time: float):
        """Сохраняет информацию о построении индекса"""
        info = {
            "model": self.embedding_model._modules['0'].auto_model.config.name_or_path,
            "embedding_size": self.embedding_size,
            "vector_db": self.vector_db_type,
            "input_folder": input_folder,
            "output_path": output_path,
            "num_chunks": num_chunks,
            "build_time_seconds": total_time,
            "build_time_minutes": total_time / 60,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        info_path = Path(output_path).parent / "build_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        print(f"Информация о построении сохранена в {info_path}")

def main():
    INPUT_FOLDER = "../knowledge_base"
    OUTPUT_PATH = "./vector_index"
    
    index_builder = VectorIndexBuilder(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        vector_db_type="chroma"
    )
    index = index_builder.build_index(
        input_folder=INPUT_FOLDER,
        output_path=OUTPUT_PATH,
        chunk_size=300, 
        chunk_overlap=50
    )

if __name__ == "__main__":
    main()