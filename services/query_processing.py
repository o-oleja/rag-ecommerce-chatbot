"""
Модуль для обработки запросов пользователя (Блок 3)
Обрабатывает вопросы пользователей с помощью RAG и LLM
"""

import logging
import re
from typing import Dict, Any, List, Tuple
from openai import OpenAI

from utils.embeddings import EmbeddingService
from utils.vector_storage import VectorDatabase
from utils.text_processing import prepare_text_for_embedding


class QueryProcessor:
    """
    Процессор запросов для RAG системы
    Обрабатывает вопросы пользователей и генерирует ответы
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.qp_config = config['query_processing']
        self.openai_config = config['openai']
        
        # Инициализируем OpenAI клиент
        self.client = OpenAI(
            api_key=self.openai_config['api_key'],
            timeout=self.openai_config.get('timeout', 30)
        )
        
        # Инициализируем сервис эмбеддингов
        self.embedding_service = EmbeddingService(
            client=self.client,
            model=self.qp_config['embedding_model']
        )
        
        logging.info("QueryProcessor инициализирован")
        
    def process_query(self, question: str, vector_db: VectorDatabase) -> Dict[str, Any]:
        """
        Обрабатывает запрос пользователя полным RAG пайплайном
        
        Args:
            question (str): Вопрос пользователя
            vector_db (VectorDatabase): Векторная база знаний
            
        Returns:
            Dict[str, Any]: Результат обработки с ответом и метаданными
        """
        logging.info(f"Обрабатываем запрос: {question}")
        
        try:
            # 1. Создаем эмбеддинг для вопроса
            query_embedding = self._create_query_embedding(question)
            
            # 2. Ищем релевантные документы
            relevant_docs = self._search_relevant_documents(
                query_embedding, vector_db
            )
            
            # 3. Подготавливаем контекст
            context = self._prepare_context(relevant_docs)
            
            # 4. Генерируем ответ
            answer = self._generate_answer(question, context)
            
            # 5. Оцениваем качество ответа
            evaluation_score = self._evaluate_response(question, answer, context)
            
            result = {
                'question': question,
                'answer': answer,
                'context': context,
                'relevant_documents': relevant_docs,
                'evaluation_score': evaluation_score,
                'success': True
            }
            
            logging.info(f"Запрос обработан успешно. Оценка: {evaluation_score}")
            return result
            
        except Exception as e:
            logging.error(f"Ошибка обработки запроса: {e}")
            return {
                'question': question,
                'answer': "Извините, произошла ошибка при обработке вашего вопроса.",
                'context': "",
                'relevant_documents': [],
                'evaluation_score': 0.0,
                'success': False,
                'error': str(e)
            }
            
    def _create_query_embedding(self, question: str) -> List[float]:
        """Создает эмбеддинг для вопроса пользователя"""
        prepared_question = prepare_text_for_embedding(question)
        return self.embedding_service.create_embedding(prepared_question)
        
    def _search_relevant_documents(self, query_embedding: List[float], vector_db: VectorDatabase) -> List[Tuple[str, float]]:
        """Ищет релевантные документы в векторной базе"""
        k = self.qp_config['top_k_results']
        return vector_db.search(query_embedding, k=k)
        
    def _prepare_context(self, relevant_docs: List[Tuple[str, float]]) -> str:
        """Подготавливает контекст из релевантных документов"""
        if not relevant_docs:
            return ""
            
        # Собираем тексты из релевантных документов
        texts = [doc[0] for doc in relevant_docs]
        context = "\n\n".join(texts)
        
        # Обрезаем контекст если он слишком длинный
        max_length = self.qp_config['max_context_length']
        if len(context) > max_length:
            context = context[:max_length]
            # Обрезаем по последнему полному предложению
            last_period = context.rfind('.')
            if last_period > max_length * 0.8:  # Если точка не слишком далеко
                context = context[:last_period + 1]
            else:
                context += "..."
                
        return context
        
    def _generate_answer(self, question: str, context: str) -> str:
        """Генерирует ответ используя LLM"""
        system_prompt = self.qp_config['system_prompt']
        
        # Формируем промпт для пользователя
        user_prompt = self.qp_config['user_prompt_template'].format(
            context=context,
            question=question
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.qp_config['llm_model'],
                temperature=self.openai_config.get('temperature', 0),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Ошибка генерации ответа: {e}")
            raise
            
    def _evaluate_response(self, question: str, answer: str, context: str) -> float:
        """Оценивает качество ответа"""
        try:
            evaluation_prompt = self.qp_config['evaluation_prompt'].format(
                question=question,
                answer=answer,
                context=context
            )
            
            response = self.client.chat.completions.create(
                model=self.qp_config['llm_model'],
                temperature=0,
                messages=[
                    {"role": "user", "content": evaluation_prompt}
                ]
            )
            
            evaluation_text = response.choices[0].message.content.strip()
            
            # Извлекаем число от 0 до 10
            score_match = re.search(r'\b(\d{1,2}(?:\.\d+)?)\b', evaluation_text)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 10.0)  # Ограничиваем диапазон 0-10
            else:
                logging.warning(f"Не удалось извлечь оценку из: {evaluation_text}")
                return 5.0  # Средняя оценка по умолчанию
                
        except Exception as e:
            logging.error(f"Ошибка оценки ответа: {e}")
            return 5.0  # Средняя оценка в случае ошибки
            
    def search_documents(self, query: str, vector_db: VectorDatabase, k: int = None) -> List[Tuple[str, float]]:
        """
        Ищет документы по запросу без генерации ответа
        
        Args:
            query (str): Поисковый запрос
            vector_db (VectorDatabase): Векторная база
            k (int): Количество результатов (по умолчанию из конфига)
            
        Returns:
            List[Tuple[str, float]]: Список документов с оценками релевантности
        """
        if k is None:
            k = self.qp_config['top_k_results']
            
        query_embedding = self._create_query_embedding(query)
        return vector_db.search(query_embedding, k=k)
        
    def generate_answer_from_context(self, question: str, context: str) -> str:
        """
        Генерирует ответ для заданного контекста без поиска
        
        Args:
            question (str): Вопрос пользователя
            context (str): Готовый контекст
            
        Returns:
            str: Сгенерированный ответ
        """
        return self._generate_answer(question, context)
        
    def batch_process_queries(self, questions: List[str], vector_db: VectorDatabase) -> List[Dict[str, Any]]:
        """
        Обрабатывает несколько запросов пакетом
        
        Args:
            questions (List[str]): Список вопросов
            vector_db (VectorDatabase): Векторная база знаний
            
        Returns:
            List[Dict[str, Any]]: Список результатов обработки
        """
        results = []
        
        for i, question in enumerate(questions):
            logging.info(f"Обрабатываем вопрос {i+1}/{len(questions)}")
            result = self.process_query(question, vector_db)
            results.append(result)
            
        return results
        
    def get_processing_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Вычисляет статистику по результатам обработки
        
        Args:
            results (List[Dict[str, Any]]): Результаты обработки запросов
            
        Returns:
            Dict[str, Any]: Статистика
        """
        if not results:
            return {}
            
        successful_results = [r for r in results if r.get('success', False)]
        
        evaluation_scores = [r['evaluation_score'] for r in successful_results if 'evaluation_score' in r]
        
        stats = {
            'total_queries': len(results),
            'successful_queries': len(successful_results),
            'failed_queries': len(results) - len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'average_evaluation_score': sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0,
            'min_evaluation_score': min(evaluation_scores) if evaluation_scores else 0,
            'max_evaluation_score': max(evaluation_scores) if evaluation_scores else 0
        }
        
        return stats 