"""
Тесты для утилит RAG системы
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from utils.text_processing import (
    chunk_text, clean_text, extract_qa_content, prepare_text_for_embedding
)
from utils.embeddings import (
    EmbeddingService, cosine_similarity, euclidean_distance, 
    dot_product_similarity, get_similarity_function
)
from utils.vector_storage import (
    MemoryVectorDatabase, PickleVectorDatabase, create_vector_database
)


class TestTextProcessing:
    """Тесты для утилит обработки текста"""
    
    def test_chunk_text_basic(self):
        """Тест базового чанкирования текста"""
        text = "Это тестовый текст для проверки функции чанкирования. Он содержит несколько предложений."
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 40 for chunk in chunks)  # С учетом overlap
        
    def test_chunk_text_short(self):
        """Тест чанкирования короткого текста"""
        text = "Короткий текст"
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text
        
    def test_chunk_text_empty(self):
        """Тест чанкирования пустого текста"""
        chunks = chunk_text("", chunk_size=50, overlap=10)
        assert chunks == []
        
        chunks = chunk_text("   ", chunk_size=50, overlap=10)
        assert chunks == []
        
    def test_chunk_text_invalid_overlap(self):
        """Тест чанкирования с неверным перекрытием"""
        with pytest.raises(ValueError, match="Перекрытие не может быть больше"):
            chunk_text("текст", chunk_size=10, overlap=15)
            
    def test_clean_text(self):
        """Тест очистки текста"""
        text = "  Текст   с    множественными    пробелами  \n\n  "
        cleaned = clean_text(text)
        
        assert cleaned == "Текст с множественными пробелами"
        
    def test_clean_text_empty(self):
        """Тест очистки пустого текста"""
        assert clean_text("") == ""
        assert clean_text(None) == ""
        
    def test_extract_qa_content(self):
        """Тест извлечения контента из вопрос-ответ"""
        qa_item = {
            'question': 'Тестовый вопрос?',
            'answer': 'Тестовый ответ.'
        }
        
        content = extract_qa_content(qa_item)
        expected = "Вопрос: Тестовый вопрос?\nОтвет: Тестовый ответ."
        
        assert content == expected
        
    def test_extract_qa_content_only_answer(self):
        """Тест извлечения контента только с ответом"""
        qa_item = {'answer': 'Только ответ.'}
        content = extract_qa_content(qa_item)
        
        assert content == "Только ответ."
        
    def test_extract_qa_content_empty(self):
        """Тест извлечения пустого контента"""
        qa_item = {}
        content = extract_qa_content(qa_item)
        
        assert content == ""
        
    def test_prepare_text_for_embedding(self):
        """Тест подготовки текста для эмбеддинга"""
        text = "  Текст   с    пробелами  \n\n  "
        prepared = prepare_text_for_embedding(text)
        
        assert prepared == "Текст с пробелами"
        
    def test_prepare_text_for_embedding_long(self):
        """Тест подготовки длинного текста для эмбеддинга"""
        long_text = "А" * 10000  # Очень длинный текст
        prepared = prepare_text_for_embedding(long_text)
        
        assert len(prepared) <= 8000


class TestEmbeddings:
    """Тесты для утилит работы с эмбеддингами"""
    
    def test_embedding_service_init(self):
        """Тест инициализации сервиса эмбеддингов"""
        mock_client = Mock()
        service = EmbeddingService(mock_client, "test-model")
        
        assert service.client == mock_client
        assert service.model == "test-model"
        
    def test_create_embedding(self):
        """Тест создания эмбеддинга"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response
        
        service = EmbeddingService(mock_client)
        embedding = service.create_embedding("тестовый текст")
        
        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()
        
    def test_create_embedding_empty_text(self):
        """Тест создания эмбеддинга для пустого текста"""
        mock_client = Mock()
        service = EmbeddingService(mock_client)
        
        with pytest.raises(ValueError, match="Текст не может быть пустым"):
            service.create_embedding("")
            
    def test_create_embeddings_batch(self):
        """Тест создания эмбеддингов пакетом"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response
        
        service = EmbeddingService(mock_client)
        texts = ["текст 1", "текст 2"]
        embeddings = service.create_embeddings_batch(texts)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.1, 0.2, 0.3]
        
    def test_cosine_similarity(self):
        """Тест вычисления косинусного сходства"""
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        vec3 = [1, 0, 0]
        
        # Перпендикулярные векторы
        assert abs(cosine_similarity(vec1, vec2) - 0.0) < 1e-6
        
        # Одинаковые векторы
        assert abs(cosine_similarity(vec1, vec3) - 1.0) < 1e-6
        
    def test_cosine_similarity_zero_vector(self):
        """Тест косинусного сходства с нулевым вектором"""
        vec1 = [1, 2, 3]
        vec_zero = [0, 0, 0]
        
        similarity = cosine_similarity(vec1, vec_zero)
        assert similarity == 0.0
        
    def test_euclidean_distance(self):
        """Тест вычисления евклидова расстояния"""
        vec1 = [0, 0, 0]
        vec2 = [3, 4, 0]
        
        distance = euclidean_distance(vec1, vec2)
        assert abs(distance - 5.0) < 1e-6  # 3-4-5 треугольник
        
    def test_dot_product_similarity(self):
        """Тест вычисления скалярного произведения"""
        vec1 = [1, 2, 3]
        vec2 = [4, 5, 6]
        
        product = dot_product_similarity(vec1, vec2)
        expected = 1*4 + 2*5 + 3*6  # = 32
        assert product == expected
        
    def test_get_similarity_function(self):
        """Тест получения функции схожести"""
        cosine_func = get_similarity_function('cosine')
        euclidean_func = get_similarity_function('euclidean')
        dot_func = get_similarity_function('dot_product')
        
        assert callable(cosine_func)
        assert callable(euclidean_func)
        assert callable(dot_func)
        
        with pytest.raises(ValueError, match="Неподдерживаемый тип"):
            get_similarity_function('unknown')


class TestVectorStorage:
    """Тесты для векторного хранилища"""
    
    def test_memory_vector_database_init(self):
        """Тест инициализации векторной базы в памяти"""
        db = MemoryVectorDatabase('cosine')
        
        assert db.texts == []
        assert db.embeddings == []
        assert db.metadata['similarity_function'] == 'cosine'
        
    def test_add_documents(self):
        """Тест добавления документов в базу"""
        db = MemoryVectorDatabase()
        
        texts = ["документ 1", "документ 2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        db.add_documents(texts, embeddings)
        
        assert db.texts == texts
        assert db.embeddings == embeddings
        
    def test_add_documents_mismatch(self):
        """Тест добавления документов с несоответствием количества"""
        db = MemoryVectorDatabase()
        
        texts = ["документ 1", "документ 2"]
        embeddings = [[0.1, 0.2]]  # Меньше эмбеддингов
        
        with pytest.raises(ValueError, match="Количество текстов должно соответствовать"):
            db.add_documents(texts, embeddings)
            
    def test_search(self):
        """Тест поиска в векторной базе"""
        db = MemoryVectorDatabase()
        
        texts = ["документ 1", "документ 2", "документ 3"]
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        db.add_documents(texts, embeddings)
        
        query_embedding = [1.0, 0.0]  # Ближе к первому документу
        results = db.search(query_embedding, k=2)
        
        assert len(results) == 2
        assert results[0][0] == "документ 1"  # Самый похожий
        assert results[0][1] > results[1][1]  # Убывающий порядок схожести
        
    def test_search_empty_db(self):
        """Тест поиска в пустой базе"""
        db = MemoryVectorDatabase()
        results = db.search([1.0, 0.0], k=5)
        
        assert results == []
        
    def test_get_stats(self):
        """Тест получения статистики базы"""
        db = MemoryVectorDatabase()
        
        texts = ["документ 1", "документ 2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        db.add_documents(texts, embeddings)
        
        stats = db.get_stats()
        
        assert stats['document_count'] == 2
        assert stats['embedding_count'] == 2
        assert stats['similarity_function'] == 'cosine'
        
    def test_create_vector_database_memory(self):
        """Тест создания векторной базы в памяти"""
        db = create_vector_database('memory', similarity_function='cosine')
        
        assert isinstance(db, MemoryVectorDatabase)
        
    def test_create_vector_database_pickle(self):
        """Тест создания pickle векторной базы"""
        with patch('utils.vector_storage.Path'):
            db = create_vector_database('pickle', filepath='test.pkl')
            
            assert isinstance(db, PickleVectorDatabase)
            
    def test_create_vector_database_unknown(self):
        """Тест создания неизвестного типа базы"""
        with pytest.raises(ValueError, match="Неподдерживаемый тип"):
            create_vector_database('unknown_type') 