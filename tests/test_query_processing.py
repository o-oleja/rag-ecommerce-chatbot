"""
Тесты для модуля query_processing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from services.query_processing import QueryProcessor
from utils.vector_storage import MemoryVectorDatabase


@pytest.fixture
def mock_config():
    """Фикстура с тестовой конфигурацией"""
    return {
        'query_processing': {
            'llm_model': 'gpt-4o-mini-2024-07-18',
            'embedding_model': 'text-embedding-3-small',
            'top_k_results': 3,
            'max_context_length': 1000,
            'system_prompt': 'Вы помощник интернет-магазина.',
            'user_prompt_template': 'Контекст: {context}\nВопрос: {question}',
            'evaluation_prompt': 'Оцените ответ от 0 до 10.\nВопрос: {question}\nОтвет: {answer}\nКонтекст: {context}'
        },
        'openai': {
            'api_key': 'test-api-key',
            'timeout': 30,
            'temperature': 0
        }
    }


@pytest.fixture
def mock_vector_db():
    """Фикстура с моком векторной базы"""
    mock_db = Mock()
    mock_db.search.return_value = [
        ('Документ 1: информация о доставке', 0.95),
        ('Документ 2: информация об оплате', 0.87),
        ('Документ 3: информация о возврате', 0.76)
    ]
    return mock_db


@pytest.fixture
def mock_embedding_service():
    """Фикстура с моком сервиса эмбеддингов"""
    mock_service = Mock()
    mock_service.create_embedding.return_value = [0.1, 0.2, 0.3] * 512  # 1536-мерный вектор
    return mock_service


@pytest.fixture
def mock_openai_response():
    """Фикстура с моком ответа OpenAI"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Тестовый ответ от LLM"
    return mock_response


class TestQueryProcessor:
    """Тесты для класса QueryProcessor"""
    
    def test_init(self, mock_config):
        """Тест инициализации QueryProcessor"""
        with patch('services.query_processing.OpenAI'), \
             patch('services.query_processing.EmbeddingService'):
            
            processor = QueryProcessor(mock_config)
            
            assert processor.config == mock_config
            assert processor.qp_config == mock_config['query_processing']
            assert processor.openai_config == mock_config['openai']
            
    def test_create_query_embedding(self, mock_config, mock_embedding_service):
        """Тест создания эмбеддинга для запроса"""
        with patch('services.query_processing.OpenAI'), \
             patch('services.query_processing.EmbeddingService', return_value=mock_embedding_service):
            
            processor = QueryProcessor(mock_config)
            
            question = "Какие способы оплаты доступны?"
            embedding = processor._create_query_embedding(question)
            
            mock_embedding_service.create_embedding.assert_called_once()
            assert embedding == mock_embedding_service.create_embedding.return_value
            
    def test_search_relevant_documents(self, mock_config, mock_vector_db, mock_embedding_service):
        """Тест поиска релевантных документов"""
        with patch('services.query_processing.OpenAI'), \
             patch('services.query_processing.EmbeddingService', return_value=mock_embedding_service):
            
            processor = QueryProcessor(mock_config)
            
            query_embedding = [0.1, 0.2, 0.3] * 512
            results = processor._search_relevant_documents(query_embedding, mock_vector_db)
            
            mock_vector_db.search.assert_called_once_with(query_embedding, k=3)
            assert results == mock_vector_db.search.return_value
            
    def test_prepare_context(self, mock_config):
        """Тест подготовки контекста из документов"""
        with patch('services.query_processing.OpenAI'), \
             patch('services.query_processing.EmbeddingService'):
            
            processor = QueryProcessor(mock_config)
            
            relevant_docs = [
                ('Документ 1', 0.95),
                ('Документ 2', 0.87),
                ('Документ 3', 0.76)
            ]
            
            context = processor._prepare_context(relevant_docs)
            
            expected = "Документ 1\n\nДокумент 2\n\nДокумент 3"
            assert context == expected
            
    def test_prepare_context_long_text(self, mock_config):
        """Тест обрезки длинного контекста"""
        with patch('services.query_processing.OpenAI'), \
             patch('services.query_processing.EmbeddingService'):
            
            processor = QueryProcessor(mock_config)
            
            # Создаем длинный документ (немного меньше для учета возможного добавления "...")
            long_text = "А" * 900 + ". " + "Б" * 295 + "."
            relevant_docs = [(long_text, 0.95)]
            
            context = processor._prepare_context(relevant_docs)
            
            # Контекст должен быть обрезан до max_context_length (1000) или немного больше с "..."
            assert len(context) <= 1005  # Даем небольшую погрешность на "..."
            assert context.endswith(".") or context.endswith("...")  # Обрезан корректно
            
    def test_prepare_context_empty(self, mock_config):
        """Тест подготовки контекста из пустого списка документов"""
        with patch('services.query_processing.OpenAI'), \
             patch('services.query_processing.EmbeddingService'):
            
            processor = QueryProcessor(mock_config)
            
            context = processor._prepare_context([])
            
            assert context == ""
            
    def test_generate_answer(self, mock_config, mock_openai_response):
        """Тест генерации ответа"""
        with patch('services.query_processing.OpenAI') as mock_openai_class, \
             patch('services.query_processing.EmbeddingService'):
            
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_class.return_value = mock_client
            
            processor = QueryProcessor(mock_config)
            processor.client = mock_client
            
            question = "Тестовый вопрос?"
            context = "Тестовый контекст"
            
            answer = processor._generate_answer(question, context)
            
            assert answer == "Тестовый ответ от LLM"
            mock_client.chat.completions.create.assert_called_once()
            
    def test_evaluate_response(self, mock_config):
        """Тест оценки качества ответа"""
        with patch('services.query_processing.OpenAI') as mock_openai_class, \
             patch('services.query_processing.EmbeddingService'):
            
            # Настраиваем мок для возврата оценки
            mock_eval_response = Mock()
            mock_eval_response.choices = [Mock()]
            mock_eval_response.choices[0].message.content = "8.5"
            
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_eval_response
            mock_openai_class.return_value = mock_client
            
            processor = QueryProcessor(mock_config)
            processor.client = mock_client
            
            question = "Тестовый вопрос?"
            answer = "Тестовый ответ"
            context = "Тестовый контекст"
            
            score = processor._evaluate_response(question, answer, context)
            
            assert score == 8.5
            mock_client.chat.completions.create.assert_called_once()
            
    def test_evaluate_response_invalid_score(self, mock_config):
        """Тест оценки ответа с некорректной оценкой"""
        with patch('services.query_processing.OpenAI') as mock_openai_class, \
             patch('services.query_processing.EmbeddingService'):
            
            # Настраиваем мок для возврата некорректного ответа
            mock_eval_response = Mock()
            mock_eval_response.choices = [Mock()]
            mock_eval_response.choices[0].message.content = "Нет числовой оценки"
            
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_eval_response
            mock_openai_class.return_value = mock_client
            
            processor = QueryProcessor(mock_config)
            processor.client = mock_client
            
            score = processor._evaluate_response("вопрос", "ответ", "контекст")
            
            assert score == 5.0  # Должна вернуться средняя оценка
            
    def test_process_query_success(self, mock_config, mock_vector_db, mock_embedding_service):
        """Тест успешной обработки запроса"""
        with patch('services.query_processing.OpenAI') as mock_openai_class, \
             patch('services.query_processing.EmbeddingService', return_value=mock_embedding_service):
            
            # Настраиваем моки
            mock_answer_response = Mock()
            mock_answer_response.choices = [Mock()]
            mock_answer_response.choices[0].message.content = "Ответ на вопрос"
            
            mock_eval_response = Mock()
            mock_eval_response.choices = [Mock()]
            mock_eval_response.choices[0].message.content = "8"
            
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = [
                mock_answer_response,  # Для генерации ответа
                mock_eval_response     # Для оценки
            ]
            mock_openai_class.return_value = mock_client
            
            processor = QueryProcessor(mock_config)
            processor.client = mock_client
            
            question = "Какие способы оплаты доступны?"
            result = processor.process_query(question, mock_vector_db)
            
            assert result['success'] is True
            assert result['question'] == question
            assert result['answer'] == "Ответ на вопрос"
            assert result['evaluation_score'] == 8.0
            assert len(result['relevant_documents']) == 3
            
    def test_process_query_error(self, mock_config, mock_vector_db, mock_embedding_service):
        """Тест обработки запроса с ошибкой"""
        with patch('services.query_processing.OpenAI') as mock_openai_class, \
             patch('services.query_processing.EmbeddingService', return_value=mock_embedding_service):
            
            # Настраиваем мок для выброса исключения
            mock_embedding_service.create_embedding.side_effect = Exception("API Error")
            
            processor = QueryProcessor(mock_config)
            
            question = "Тестовый вопрос?"
            result = processor.process_query(question, mock_vector_db)
            
            assert result['success'] is False
            assert result['question'] == question
            assert result['answer'] == "Извините, произошла ошибка при обработке вашего вопроса."
            assert result['evaluation_score'] == 0.0
            assert 'error' in result
            
    def test_search_documents(self, mock_config, mock_vector_db, mock_embedding_service):
        """Тест поиска документов без генерации ответа"""
        with patch('services.query_processing.OpenAI'), \
             patch('services.query_processing.EmbeddingService', return_value=mock_embedding_service):
            
            processor = QueryProcessor(mock_config)
            
            query = "тестовый запрос"
            results = processor.search_documents(query, mock_vector_db)
            
            mock_embedding_service.create_embedding.assert_called_once()
            mock_vector_db.search.assert_called_once()
            assert results == mock_vector_db.search.return_value
            
    def test_batch_process_queries(self, mock_config, mock_vector_db, mock_embedding_service):
        """Тест пакетной обработки запросов"""
        with patch('services.query_processing.OpenAI') as mock_openai_class, \
             patch('services.query_processing.EmbeddingService', return_value=mock_embedding_service):
            
            # Настраиваем мок клиента
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "8"
            
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            processor = QueryProcessor(mock_config)
            processor.client = mock_client
            
            questions = ["Вопрос 1?", "Вопрос 2?"]
            results = processor.batch_process_queries(questions, mock_vector_db)
            
            assert len(results) == 2
            assert all(result['success'] for result in results)
            
    def test_get_processing_stats(self, mock_config):
        """Тест вычисления статистики обработки"""
        with patch('services.query_processing.OpenAI'), \
             patch('services.query_processing.EmbeddingService'):
            
            processor = QueryProcessor(mock_config)
            
            results = [
                {'success': True, 'evaluation_score': 8.0},
                {'success': True, 'evaluation_score': 9.0},
                {'success': False, 'evaluation_score': 0.0},
                {'success': True, 'evaluation_score': 7.0}
            ]
            
            stats = processor.get_processing_stats(results)
            
            assert stats['total_queries'] == 4
            assert stats['successful_queries'] == 3
            assert stats['failed_queries'] == 1
            assert stats['success_rate'] == 0.75
            assert stats['average_evaluation_score'] == 8.0  # (8+9+7)/3
            assert stats['min_evaluation_score'] == 7.0
            assert stats['max_evaluation_score'] == 9.0 