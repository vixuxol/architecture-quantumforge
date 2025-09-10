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

MODEL_NAME = "openai/gpt-oss-20b"

class RAGPipeline:
    __templates_path__ = "./templates"
    __persisted_chroma_dir__ = "../index_creation/vector_index"
    __prompt_template_name__ = "prompt.j2"
    __question_template_name__ = "question.j2"
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_db_type: str = "chroma",  # или "faiss" (как возможность будущего расширения)
                 llm_provider: str = "local"):  # или "yandexgpt", "openai"
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_db_type = vector_db_type
        self.llm_provider = llm_provider
        
        if vector_db_type == "chroma":
            self.chroma_client = PersistentClient(path=self.__persisted_chroma_dir__)
            self.collection = self.chroma_client.get_or_create_collection(name="knowledge_base")
        elif vector_db_type == "faiss":
            raise NotImplementedError("This vector_db_type wasn't implemented yet")
        else:
            raise TypeError("Unknown vector db technology")
        
        self.template_env = Environment(
            loader=FileSystemLoader(Path(self.__templates_path__)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )

        if llm_provider == "local":
            # нужен токен внутри os.environ["HF_TOKEN"]
            self.pipeline = pipeline(
                "text-generation",
                model=MODEL_NAME,
                device=-1
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

    def process_llm_response(self, raw_response: str):
        parts = raw_response.split('assistantfinal', 1)
        reasoning = parts[0]
        final_answer = parts[1] if len(parts) > 1 else raw_response

        if reasoning.startswith("analysis"):
            reasoning = reasoning[8:].lstrip()

        return {
            "reasoning": reasoning.strip(),
            "final_answer": final_answer.strip()
        }
    
    def call_llm(self, prompt: str, question: str) -> str:
        """Отправляет запрос в LLM"""
        
        if self.llm_provider == "openai":
            # API ключ: os.environ["OPENAI_API_KEY"] = "your-key"
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
            messages = [
                 {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ]
            outputs = self.pipeline(messages)
            raw_result = outputs[0]["generated_text"][-1]['content']
            result = self.process_llm_response(raw_result)

            return f'Reasoning:\n{result["reasoning"]}'+"\n" + "-" * 50 + "\n"+f'Final Answer::\n{result["final_answer"]}'
        return "LLM provider not implemented"
    
    def process_query(self, query: str, k: int = 5) -> str:
        """Полный пайплайн обработки запроса"""
        print(f"\nProcessing user query: {query}\n")
        print("=" * 80)
        
        # Поиск релевантных чанков
        similar_chunks = self.search_similar_chunks(query, k)
        print(f"📄 Were found {len(similar_chunks)} relevant fragments\n")
        
        # Создание промпта
        prompt, question = self.create_prompt_and_question(query, similar_chunks)
        print("📝 Generated prompt to LLM:\n")
        print(prompt)
        print(question)
        print("=" * 80)
        
        # # Вызов LLM
        print("⭐ Result:\n")
        response = self.call_llm(prompt, question)
        print(response)
        
        print("=" * 80)

def main():
    rag = RAGPipeline(vector_db_type="chroma", llm_provider="local")
    
    print("🔍 Interactive mode of operation")
    print("Enter a request (or 'quit' to exit):")
    
    while True:
        try:
            query = input("\nEnter user question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            rag.process_query(query)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Search error: {e}")

if __name__ == "__main__":
    main()