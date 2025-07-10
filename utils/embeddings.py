"""
Утилиты для работы с эмбеддингами в RAG системе
"""

import numpy as np
import logging
from typing import List, Union
from openai import OpenAI


class EmbeddingService:
    """Сервис для создания и работы с эмбеддингами"""
    
    def __init__(self, client: OpenAI, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model
        
    def create_embedding(self, text: str) -> List[float]:
        """
        Создает эмбеддинг для текста
        
        Args:
            text (str): Текст для создания эмбеддинга
            
        Returns:
            List[float]: Вектор эмбеддинга
            
        Raises:
            Exception: При ошибке создания эмбеддинга
        """
        try:
            if not text or not text.strip():
                raise ValueError("Текст не может быть пустым")
                
            response = self.client.embeddings.create(
                model=self.model,
                input=text.strip()
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logging.error(f"Ошибка создания эмбеддинга: {e}")
            raise
            
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги для списка текстов
        
        Args:
            texts (List[str]): Список текстов
            
        Returns:
            List[List[float]]: Список векторов эмбеддингов
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                if i % 10 == 0:
                    logging.info(f"Создание эмбеддингов: {i}/{len(texts)}")
                    
                embedding = self.create_embedding(text)
                embeddings.append(embedding)
                
            except Exception as e:
                logging.error(f"Ошибка создания эмбеддинга для текста {i}: {e}")
                # Добавляем нулевой вектор в случае ошибки
                embeddings.append([0.0] * 1536)  # Размерность для text-embedding-3-small
                continue
                
        logging.info(f"Создано {len(embeddings)} эмбеддингов")
        return embeddings


def cosine_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """
    Вычисляет косинусное сходство между двумя векторами
    
    Args:
        vec1: Первый вектор
        vec2: Второй вектор
        
    Returns:
        float: Значение косинусного сходства от -1 до 1
    """
    # Преобразуем в numpy массивы
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    
    # Проверяем, что векторы не нулевые
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    # Вычисляем косинусное сходство
    similarity = np.dot(v1, v2) / (norm1 * norm2)
    
    # Ограничиваем значение в диапазоне [-1, 1] из-за возможных погрешностей вычислений
    return float(np.clip(similarity, -1.0, 1.0))


def euclidean_distance(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """
    Вычисляет евклидово расстояние между двумя векторами
    
    Args:
        vec1: Первый вектор
        vec2: Второй вектор
        
    Returns:
        float: Евклидово расстояние
    """
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    
    return float(np.linalg.norm(v1 - v2))


def dot_product_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """
    Вычисляет скалярное произведение двух векторов (как меру схожести)
    
    Args:
        vec1: Первый вектор
        vec2: Второй вектор
        
    Returns:
        float: Скалярное произведение
    """
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    
    return float(np.dot(v1, v2))


def get_similarity_function(similarity_type: str):
    """
    Возвращает функцию для вычисления схожести по её типу
    
    Args:
        similarity_type (str): Тип функции схожести ('cosine', 'euclidean', 'dot_product')
        
    Returns:
        callable: Функция для вычисления схожести
        
    Raises:
        ValueError: Если указан неподдерживаемый тип функции
    """
    similarity_functions = {
        'cosine': cosine_similarity,
        'euclidean': lambda v1, v2: -euclidean_distance(v1, v2),  # Отрицательное расстояние для сортировки
        'dot_product': dot_product_similarity
    }
    
    if similarity_type not in similarity_functions:
        raise ValueError(f"Неподдерживаемый тип функции схожести: {similarity_type}")
        
    return similarity_functions[similarity_type] 