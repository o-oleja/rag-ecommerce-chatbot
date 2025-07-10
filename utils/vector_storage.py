"""
Утилиты для векторного хранилища в RAG системе
Поддерживает разные типы векторных баз
"""

import pickle
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from datetime import datetime
from pathlib import Path

from .embeddings import get_similarity_function


class VectorDatabase(ABC):
    """Абстрактный класс для векторной базы данных"""
    
    @abstractmethod
    def add_documents(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Добавляет документы и их эмбеддинги в базу"""
        pass
        
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Ищет наиболее похожие документы"""
        pass
        
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Сохраняет базу в файл"""
        pass
        
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Загружает базу из файла"""
        pass
        
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику базы"""
        pass


class MemoryVectorDatabase(VectorDatabase):
    """Векторная база данных в памяти с простым линейным поиском"""
    
    def __init__(self, similarity_function: str = "cosine"):
        self.texts = []
        self.embeddings = []
        self.similarity_func = get_similarity_function(similarity_function)
        self.metadata = {
            'created_at': datetime.now(),
            'similarity_function': similarity_function
        }
        
    def add_documents(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Добавляет документы и их эмбеддинги в базу"""
        if len(texts) != len(embeddings):
            raise ValueError("Количество текстов должно соответствовать количеству эмбеддингов")
            
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        
        logging.info(f"Добавлено {len(texts)} документов в векторную базу")
        
    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Ищет наиболее похожие документы"""
        if not self.embeddings:
            return []
            
        similarities = []
        
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.similarity_func(query_embedding, doc_embedding)
            similarities.append((i, similarity))
            
        # Сортируем по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-k результатов
        results = []
        for i, similarity in similarities[:k]:
            results.append((self.texts[i], similarity))
            
        return results
        
    def save(self, filepath: str) -> None:
        """Сохраняет базу в файл"""
        try:
            data = {
                'texts': self.texts,
                'embeddings': self.embeddings,
                'metadata': {
                    **self.metadata,
                    'saved_at': datetime.now(),
                    'document_count': len(self.texts)
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
            logging.info(f"Векторная база сохранена в {filepath}")
            
        except Exception as e:
            logging.error(f"Ошибка сохранения векторной базы: {e}")
            raise
            
    def load(self, filepath: str) -> None:
        """Загружает базу из файла"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.texts = data['texts']
            self.embeddings = data['embeddings']
            self.metadata = data.get('metadata', {})
            
            logging.info(f"Векторная база загружена из {filepath}")
            logging.info(f"Загружено {len(self.texts)} документов")
            
        except FileNotFoundError:
            logging.warning(f"Файл векторной базы не найден: {filepath}")
            raise
        except Exception as e:
            logging.error(f"Ошибка загрузки векторной базы: {e}")
            raise
            
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику базы"""
        return {
            'document_count': len(self.texts),
            'embedding_count': len(self.embeddings),
            'similarity_function': self.metadata.get('similarity_function', 'unknown'),
            'created_at': self.metadata.get('created_at'),
            'last_saved': self.metadata.get('saved_at')
        }


class PickleVectorDatabase(MemoryVectorDatabase):
    """Векторная база данных с автоматическим сохранением в pickle файл"""
    
    def __init__(self, filepath: str, similarity_function: str = "cosine"):
        super().__init__(similarity_function)
        self.filepath = Path(filepath)
        
        # Пытаемся загрузить существующую базу
        if self.filepath.exists():
            try:
                self.load(str(self.filepath))
            except Exception as e:
                logging.warning(f"Не удалось загрузить существующую базу: {e}")
                
    def add_documents(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Добавляет документы и автоматически сохраняет"""
        super().add_documents(texts, embeddings)
        self.save(str(self.filepath))


def create_vector_database(db_type: str, **kwargs) -> VectorDatabase:
    """
    Фабричная функция для создания векторной базы данных
    
    Args:
        db_type (str): Тип базы данных ('memory' или 'pickle')
        **kwargs: Дополнительные параметры для конкретного типа базы
        
    Returns:
        VectorDatabase: Экземпляр векторной базы данных
        
    Raises:
        ValueError: Если указан неподдерживаемый тип базы
    """
    if db_type == "memory":
        similarity_function = kwargs.get('similarity_function', 'cosine')
        return MemoryVectorDatabase(similarity_function)
        
    elif db_type == "pickle":
        filepath = kwargs.get('filepath', 'vector_db.pkl')
        similarity_function = kwargs.get('similarity_function', 'cosine')
        return PickleVectorDatabase(filepath, similarity_function)
        
    else:
        raise ValueError(f"Неподдерживаемый тип векторной базы: {db_type}")


def check_vector_database_exists(filepath: str) -> bool:
    """Проверяет существование файла векторной базы"""
    return Path(filepath).exists() 