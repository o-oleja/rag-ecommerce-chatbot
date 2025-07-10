"""
Тесты для модуля knowledge_processing
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from services.knowledge_processing import KnowledgeProcessor
from utils.vector_storage import MemoryVectorDatabase


@pytest.fixture
def mock_config():
    """Фикстура с тестовой конфигурацией"""
    return {
        'knowledge_processing': {
            'chunk_size': 100,
            'chunk_overlap': 20,
            'embedding_model': 'text-embedding-3-small',
            'vector_db_type': 'memory',
            'vector_db_path': 'test_vector_db.pkl',
            'similarity_function': 'cosine'
        },
        'openai': {
            'api_key': 'test-api-key',
            'timeout': 30
        }
    }


@pytest.fixture
def sample_qa_data():
    """Фикстура с тестовыми данными вопрос-ответ"""
    return [
        {
            'question': 'Какие способы оплаты доступны?',
            'answer': 'Мы принимаем банковские карты Visa, MasterCard, наличные при получении.'
        },
        {
            'question': 'Сколько времени занимает доставка?',
            'answer': 'Доставка по Москве осуществляется в течение 1-2 рабочих дней.'
        },
        {
            'question': 'Можно ли вернуть товар?',
            'answer': 'Да, вы можете вернуть товар в течение 14 дней с момента получения.'
        }
    ]


@pytest.fixture
def temp_qa_file(sample_qa_data):
    """Фикстура создающая временный JSON файл с тестовыми данными"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(sample_qa_data, f, ensure_ascii=False, indent=2)
        temp_path = f.name
    
    yield temp_path
    
    # Очистка после теста
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_embedding_service():
    """Фикстура с моком сервиса эмбеддингов"""
    mock_service = Mock()
    mock_service.create_embeddings_batch.return_value = [
        [0.1, 0.2, 0.3] * 512,  # Имитация эмбеддинга размерности 1536
        [0.2, 0.3, 0.4] * 512,
        [0.3, 0.4, 0.5] * 512
    ]
    return mock_service


class TestKnowledgeProcessor:
    """Тесты для класса KnowledgeProcessor"""
    
    def test_init(self, mock_config):
        """Тест инициализации KnowledgeProcessor"""
        with patch('services.knowledge_processing.OpenAI'), \
             patch('services.knowledge_processing.EmbeddingService'):
            
            processor = KnowledgeProcessor(mock_config)
            
            assert processor.config == mock_config
            assert processor.kp_config == mock_config['knowledge_processing']
            assert processor.openai_config == mock_config['openai']
            
    def test_load_qa_data_success(self, mock_config, temp_qa_file, sample_qa_data):
        """Тест успешной загрузки данных вопрос-ответ"""
        with patch('services.knowledge_processing.OpenAI'), \
             patch('services.knowledge_processing.EmbeddingService'):
            
            processor = KnowledgeProcessor(mock_config)
            data = processor.load_qa_data(temp_qa_file)
            
            assert len(data) == 3
            assert data == sample_qa_data
            
    def test_load_qa_data_file_not_found(self, mock_config):
        """Тест загрузки несуществующего файла"""
        with patch('services.knowledge_processing.OpenAI'), \
             patch('services.knowledge_processing.EmbeddingService'):
            
            processor = KnowledgeProcessor(mock_config)
            
            with pytest.raises(FileNotFoundError):
                processor.load_qa_data('nonexistent_file.json')
                
    def test_load_qa_data_invalid_format(self, mock_config):
        """Тест загрузки файла с неверным форматом"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump({'invalid': 'format'}, f)  # Не список
            temp_path = f.name
            
        try:
            with patch('services.knowledge_processing.OpenAI'), \
                 patch('services.knowledge_processing.EmbeddingService'):
                
                processor = KnowledgeProcessor(mock_config)
                
                with pytest.raises(ValueError, match="JSON файл должен содержать список"):
                    processor.load_qa_data(temp_path)
        finally:
            os.unlink(temp_path)
            
    def test_load_qa_data_missing_fields(self, mock_config):
        """Тест загрузки данных с отсутствующими полями"""
        invalid_data = [{'question': 'test'}]  # Нет поля 'answer'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(invalid_data, f)
            temp_path = f.name
            
        try:
            with patch('services.knowledge_processing.OpenAI'), \
                 patch('services.knowledge_processing.EmbeddingService'):
                
                processor = KnowledgeProcessor(mock_config)
                
                with pytest.raises(ValueError, match="должен содержать 'question' и 'answer'"):
                    processor.load_qa_data(temp_path)
        finally:
            os.unlink(temp_path)
            
    def test_process_knowledge_base(self, mock_config, sample_qa_data, mock_embedding_service):
        """Тест обработки базы знаний"""
        with patch('services.knowledge_processing.OpenAI'), \
             patch('services.knowledge_processing.EmbeddingService', return_value=mock_embedding_service), \
             patch('services.knowledge_processing.create_vector_database') as mock_create_db:
            
            mock_vector_db = Mock()
            mock_create_db.return_value = mock_vector_db
            
            processor = KnowledgeProcessor(mock_config)
            result = processor.process_knowledge_base(sample_qa_data)
            
            # Проверяем, что эмбеддинги были созданы
            mock_embedding_service.create_embeddings_batch.assert_called_once()
            
            # Проверяем, что векторная база была создана
            mock_create_db.assert_called_once()
            
            # Проверяем, что документы были добавлены в базу
            mock_vector_db.add_documents.assert_called_once()
            
            assert result == mock_vector_db
            
    def test_get_or_create_vector_database_existing(self, mock_config, temp_qa_file):
        """Тест загрузки существующей векторной базы"""
        with patch('services.knowledge_processing.OpenAI'), \
             patch('services.knowledge_processing.EmbeddingService'), \
             patch('services.knowledge_processing.Path') as mock_path, \
             patch('services.knowledge_processing.create_vector_database') as mock_create_db:
            
            # Настраиваем мок для существующего файла
            mock_path.return_value.exists.return_value = True
            
            mock_vector_db = Mock()
            mock_vector_db.get_stats.return_value = {'document_count': 10}
            mock_create_db.return_value = mock_vector_db
            
            processor = KnowledgeProcessor(mock_config)
            result = processor.get_or_create_vector_database(temp_qa_file)
            
            assert result == mock_vector_db
            
    def test_get_or_create_vector_database_new(self, mock_config, sample_qa_data, mock_embedding_service):
        """Тест создания новой векторной базы"""
        with patch('services.knowledge_processing.OpenAI'), \
             patch('services.knowledge_processing.EmbeddingService', return_value=mock_embedding_service), \
             patch('services.knowledge_processing.Path') as mock_path, \
             patch('services.knowledge_processing.create_vector_database') as mock_create_db:
            
            # Настраиваем мок для несуществующего файла
            mock_path.return_value.exists.return_value = False
            
            mock_vector_db = Mock()
            mock_create_db.return_value = mock_vector_db
            
            processor = KnowledgeProcessor(mock_config)
            
            # Патчим load_qa_data для возврата тестовых данных
            with patch.object(processor, 'load_qa_data', return_value=sample_qa_data):
                result = processor.get_or_create_vector_database('test_file.json')
            
            # Проверяем, что была создана новая база
            mock_embedding_service.create_embeddings_batch.assert_called_once()
            mock_vector_db.add_documents.assert_called_once()
            
            assert result == mock_vector_db
            
    def test_validate_vector_database_success(self, mock_config):
        """Тест успешной валидации векторной базы"""
        with patch('services.knowledge_processing.OpenAI'), \
             patch('services.knowledge_processing.EmbeddingService'):
            
            processor = KnowledgeProcessor(mock_config)
            
            mock_vector_db = Mock()
            mock_vector_db.get_stats.return_value = {
                'document_count': 10,
                'embedding_count': 10
            }
            mock_vector_db.search.return_value = [('test document', 0.8)]
            
            result = processor.validate_vector_database(mock_vector_db)
            
            assert result is True
            
    def test_validate_vector_database_empty(self, mock_config):
        """Тест валидации пустой векторной базы"""
        with patch('services.knowledge_processing.OpenAI'), \
             patch('services.knowledge_processing.EmbeddingService'):
            
            processor = KnowledgeProcessor(mock_config)
            
            mock_vector_db = Mock()
            mock_vector_db.get_stats.return_value = {
                'document_count': 0,
                'embedding_count': 0
            }
            
            result = processor.validate_vector_database(mock_vector_db)
            
            assert result is False
            
    def test_validate_vector_database_mismatch(self, mock_config):
        """Тест валидации базы с несоответствием количества документов и эмбеддингов"""
        with patch('services.knowledge_processing.OpenAI'), \
             patch('services.knowledge_processing.EmbeddingService'):
            
            processor = KnowledgeProcessor(mock_config)
            
            mock_vector_db = Mock()
            mock_vector_db.get_stats.return_value = {
                'document_count': 10,
                'embedding_count': 5  # Несоответствие
            }
            
            result = processor.validate_vector_database(mock_vector_db)
            
            assert result is False 