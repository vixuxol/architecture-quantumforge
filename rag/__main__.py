import os
import numpy as np
from jinja2 import Environment, FileSystemLoader, Template
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import openai
from chromadb import PersistentClient
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
import torch

MODEL_NAME = "openai/gpt-oss-120b"

class RAGPipeline:
    __templates_path__ = "./templates"
    __persisted_chroma_dir__ = "../index_creation/vector_index"
    __prompt_template_name__ = "prompt.j2"
    __question_template_name__ = "question.j2"
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_db_type: str = "chroma",  # или "faiss" (как возможность будущего расширения)
                 llm_provider: str = "openai"):  # или "yandexgpt", "local"
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_db_type = vector_db_type
        self.llm_provider = llm_provider
        
        if vector_db_type == "chroma":
            self.chroma_client = PersistentClient(path=self.__persisted_chroma_dir__)
            self.collection = self.chroma_client.get_or_create_collection(name="knowledge_base")
        elif vector_db_type == "faiss":
            raise NotImplementedError("This vector_db_type wasn't implemented yet")
        else:
            raise TypeError("unknown vector db technology")
        
        self.template_env = Environment(
            loader=FileSystemLoader(Path(self.__templates_path__)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )

        if llm_provider == "local":
            # нужен токен внутри os.environ["HF_TOKEN"]
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype="auto",
                device_map="auto",
            )
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model.encode(texts)
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[str]:
        """Ищет k наиболее похожих чанков"""
        query_embedding = self.create_embeddings([query])
        
        if self.vector_db_type == "chroma":
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k
            )
            return results['documents'][0]
    
    def create_prompt_and_question(self, query: str, context_chunks: List[str]) -> tuple[str, str]:
        """Создает промпт с Few-shot и CoT"""
        
        prompt_template = self.template_env.get_template(self.__prompt_template_name__)
        prompt = prompt_template.render(chunks=context_chunks, question=query)

        question_template = self.template_env.get_template(self.__question_template_name__)
        question = question_template.render(chunks=context_chunks, question=query)
        
        return prompt, question
    
    def call_llm(self, prompt: str, question: str) -> str:
        """Отправляет запрос в LLM"""
        
        if self.llm_provider == "openai":
            # Установите ваш API ключ: os.environ["OPENAI_API_KEY"] = "your-key"
            client = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"]
            )
            
            response = client.responses.create(
                model="gpt-3.5-turbo",
                instructions=prompt,
                input=question
            )
            return response.output_text
        
        elif self.llm_provider == "yandexgpt":
            raise NotImplementedError("This type wasn't implemented yet")
        
        elif self.llm_provider == "local":
            formatted_prompt = f"[INST] {prompt} {question} [/INST]"
            outputs = self.generator(formatted_prompt)
            return outputs[0]["generated_text"]
        
        return "LLM provider not implemented"
    
    def process_query(self, query: str, k: int = 5) -> str:
        """Полный пайплайн обработки запроса"""
        print(f"\nОбрабатываю запрос: {query}\n")
        print("=" * 80)
        
        # Поиск релевантных чанков
        similar_chunks = self.search_similar_chunks(query, k)
        print(f"📄 Найдено {len(similar_chunks)} релевантных фрагментов\n")
        
        # Создание промпта
        prompt, question = self.create_prompt_and_question(query, similar_chunks)
        print("📝 Сформированный промпт:\n")
        print(prompt)
        print(question)
        print("=" * 80)
        
        # # Вызов LLM
        print("⭐ Результат:\n")
        response = self.call_llm(prompt, question)
        print(response)
        
        print("=" * 80)

def main():
    rag = RAGPipeline(vector_db_type="chroma", llm_provider="local")
    
    print("🔍 Интерактивный режим работы")
    print("Введите запрос (или 'quit' для выхода):")
    
    while True:
        try:
            query = input("\nВопрос: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            # работа RAG пайплайна
            rag.process_query(query)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка поиска: {e}")

if __name__ == "__main__":
    main()