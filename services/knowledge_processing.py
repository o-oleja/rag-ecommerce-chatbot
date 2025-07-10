"""
Модуль для загрузки данных и обработки знаний (Блок 2)
Обрабатывает JSON данные вопрос-ответ и создает векторную базу знаний
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from openai import OpenAI

from utils.text_processing import chunk_text, extract_qa_content, prepare_text_for_embedding
from utils.embeddings import EmbeddingService
from utils.vector_storage import create_vector_database, VectorDatabase


class KnowledgeProcessor:
    """
    Процессор знаний для RAG системы
    Загружает данные вопрос-ответ и создает векторную базу
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kp_config = config['knowledge_processing']
        self.openai_config = config['openai']
        
        # Инициализируем OpenAI клиент
        self.client = OpenAI(
            api_key=self.openai_config['api_key'],
            timeout=self.openai_config.get('timeout', 30)
        )
        
        # Инициализируем сервис эмбеддингов
        self.embedding_service = EmbeddingService(
            client=self.client,
            model=self.kp_config['embedding_model']
        )
        
        logging.info("KnowledgeProcessor инициализирован")
        
    def load_qa_data(self, filepath: str) -> List[Dict[str, str]]:
        """
        Загружает данные вопрос-ответ из JSON файла
        
        Args:
            filepath (str): Путь к JSON файлу
            
        Returns:
            List[Dict[str, str]]: Список элементов вопрос-ответ
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если файл имеет неверный формат
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                raise FileNotFoundError(f"Файл не найден: {filepath}")
                
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Валидируем формат данных
            if not isinstance(data, list):
                raise ValueError("JSON файл должен содержать список")
                
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(f"Элемент {i} должен быть объектом")
                if 'question' not in item or 'answer' not in item:
                    raise ValueError(f"Элемент {i} должен содержать 'question' и 'answer'")
                    
            logging.info(f"Загружено {len(data)} элементов вопрос-ответ из {filepath}")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON: {e}")
        except Exception as e:
            logging.error(f"Ошибка загрузки данных: {e}")
            raise
            
    def process_knowledge_base(self, qa_data: List[Dict[str, str]]) -> VectorDatabase:
        """
        Обрабатывает данные вопрос-ответ и создает векторную базу знаний
        
        Args:
            qa_data (List[Dict[str, str]]): Данные вопрос-ответ
            
        Returns:
            VectorDatabase: Векторная база знаний
        """
        logging.info("Начинаем обработку базы знаний...")
        
        # Извлекаем и подготавливаем тексты
        all_texts = []
        for item in qa_data:
            content = extract_qa_content(item)
            if content:
                all_texts.append(content)
                
        logging.info(f"Извлечено {len(all_texts)} текстов из данных")
        
        # Создаем чанки
        all_chunks = []
        chunk_size = self.kp_config['chunk_size']
        overlap = self.kp_config['chunk_overlap']
        
        for text in all_texts:
            chunks = chunk_text(text, chunk_size, overlap)
            all_chunks.extend(chunks)
            
        logging.info(f"Создано {len(all_chunks)} текстовых чанков")
        
        # Подготавливаем тексты для эмбеддинга
        prepared_chunks = []
        for chunk in all_chunks:
            prepared = prepare_text_for_embedding(chunk)
            if prepared:
                prepared_chunks.append(prepared)
                
        logging.info(f"Подготовлено {len(prepared_chunks)} чанков для эмбеддинга")
        
        # Создаем эмбеддинги
        logging.info("Создаем эмбеддинги...")
        embeddings = self.embedding_service.create_embeddings_batch(prepared_chunks)
        
        # Создаем векторную базу
        vector_db = create_vector_database(
            db_type=self.kp_config['vector_db_type'],
            filepath=self.kp_config.get('vector_db_path', 'vector_db.pkl'),
            similarity_function=self.kp_config['similarity_function']
        )
        
        # Добавляем документы в базу
        vector_db.add_documents(prepared_chunks, embeddings)
        
        logging.info("База знаний успешно создана")
        logging.info(f"Статистика: {vector_db.get_stats()}")
        
        return vector_db
        
    def get_or_create_vector_database(self, qa_filepath: str, force_rebuild: bool = False) -> VectorDatabase:
        """
        Получает существующую векторную базу или создает новую
        
        Args:
            qa_filepath (str): Путь к файлу с данными вопрос-ответ
            force_rebuild (bool): Принудительно пересоздать базу
            
        Returns:
            VectorDatabase: Векторная база знаний
        """
        db_path = self.kp_config.get('vector_db_path', 'vector_db.pkl')
        
        # Проверяем существование базы
        if not force_rebuild and Path(db_path).exists():
            try:
                logging.info(f"Загружаем существующую векторную базу из {db_path}")
                vector_db = create_vector_database(
                    db_type=self.kp_config['vector_db_type'],
                    filepath=db_path,
                    similarity_function=self.kp_config['similarity_function']
                )
                
                stats = vector_db.get_stats()
                if stats['document_count'] > 0:
                    logging.info("Векторная база успешно загружена")
                    logging.info(f"Статистика: {stats}")
                    return vector_db
                else:
                    logging.warning("Загруженная база пуста, создаем новую")
                    
            except Exception as e:
                logging.warning(f"Не удалось загрузить существующую базу: {e}")
                
        # Создаем новую базу
        logging.info("Создаем новую векторную базу...")
        qa_data = self.load_qa_data(qa_filepath)
        vector_db = self.process_knowledge_base(qa_data)
        
        return vector_db
        
    def update_knowledge_base(self, vector_db: VectorDatabase, new_qa_data: List[Dict[str, str]]) -> None:
        """
        Обновляет существующую векторную базу новыми данными
        
        Args:
            vector_db (VectorDatabase): Существующая векторная база
            new_qa_data (List[Dict[str, str]]): Новые данные вопрос-ответ
        """
        logging.info(f"Обновляем векторную базу {len(new_qa_data)} новыми элементами")
        
        # Обрабатываем новые данные так же, как и при создании базы
        new_texts = []
        for item in new_qa_data:
            content = extract_qa_content(item)
            if content:
                new_texts.append(content)
                
        # Создаем чанки
        new_chunks = []
        chunk_size = self.kp_config['chunk_size']
        overlap = self.kp_config['chunk_overlap']
        
        for text in new_texts:
            chunks = chunk_text(text, chunk_size, overlap)
            new_chunks.extend(chunks)
            
        # Подготавливаем для эмбеддинга
        prepared_chunks = []
        for chunk in new_chunks:
            prepared = prepare_text_for_embedding(chunk)
            if prepared:
                prepared_chunks.append(prepared)
                
        # Создаем эмбеддинги
        embeddings = self.embedding_service.create_embeddings_batch(prepared_chunks)
        
        # Добавляем в базу
        vector_db.add_documents(prepared_chunks, embeddings)
        
        logging.info(f"База обновлена. Новая статистика: {vector_db.get_stats()}")
        
    def validate_vector_database(self, vector_db: VectorDatabase) -> bool:
        """
        Валидирует корректность векторной базы
        
        Args:
            vector_db (VectorDatabase): Векторная база для проверки
            
        Returns:
            bool: True если база корректна
        """
        try:
            stats = vector_db.get_stats()
            
            # Проверяем базовые требования
            if stats['document_count'] == 0:
                logging.error("Векторная база пуста")
                return False
                
            if stats['document_count'] != stats['embedding_count']:
                logging.error("Количество документов не совпадает с количеством эмбеддингов")
                return False
                
            # Тестируем поиск
            test_embedding = [0.0] * 1536  # Нулевой вектор для теста
            results = vector_db.search(test_embedding, k=1)
            
            if not results:
                logging.error("Поиск в векторной базе не работает")
                return False
                
            logging.info("Векторная база прошла валидацию")
            return True
            
        except Exception as e:
            logging.error(f"Ошибка валидации векторной базы: {e}")
            return False 