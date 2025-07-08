#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG система для ответов на вопросы об интернет магазине
Использует утилиты из utils.py
"""

import json
import os
import pickle
import sys
from datetime import datetime
import utils
from openai import OpenAI
from utils import chunk_text, create_embeddings, generate_response

# Настройка OpenAI клиента (нужно установить API ключ)
client = OpenAI(
    
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")
)

# Устанавливаем клиент в модуле utils для глобального использования
utils.client = client

def load_qa_data(filename):
    """Загружает данные вопросов и ответов из JSON файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"Загружено {len(data)} пар вопрос-ответ")
        return data
    except FileNotFoundError:
        print(f"Файл {filename} не найден")
        return []
    except json.JSONDecodeError as e:
        print(f"Ошибка при чтении JSON: {e}")
        return []

def check_vector_database_exists(filename="vector_db.pkl"):
    """Проверяет существует ли файл с векторной базой"""
    return os.path.exists(filename)

def save_vector_database(text_chunks, embeddings, filename="vector_db.pkl"):
    """Сохраняет векторную базу в файл"""
    try:
        vector_db = {
            'text_chunks': text_chunks,
            'embeddings': embeddings,
            'metadata': {
                'created_at': datetime.now(),
                'chunks_count': len(text_chunks),
                'embeddings_count': len(embeddings),
                'source_file': 'ecommerce_qa.json'
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(vector_db, f)
        
        print(f"Векторная база сохранена в {filename}")
        print(f"  - Чанков: {len(text_chunks)}")
        print(f"  - Эмбеддингов: {len(embeddings)}")
        
    except Exception as e:
        print(f"Ошибка при сохранении векторной базы: {e}")

def load_vector_database(filename="vector_db.pkl"):
    """Загружает векторную базу из файла"""
    try:
        with open(filename, 'rb') as f:
            vector_db = pickle.load(f)
        
        text_chunks = vector_db['text_chunks']
        embeddings = vector_db['embeddings']
        metadata = vector_db.get('metadata', {})
        
        print(f"Векторная база загружена из {filename}")
        print(f"  - Создана: {metadata.get('created_at', 'неизвестно')}")
        print(f"  - Чанков: {metadata.get('chunks_count', len(text_chunks))}")
        print(f"  - Эмбеддингов: {metadata.get('embeddings_count', len(embeddings))}")
        
        return text_chunks, embeddings
        
    except Exception as e:
        print(f"Ошибка при загрузке векторной базы: {e}")
        return [], []

def prepare_knowledge_base(qa_data, chunk_size=500, overlap=50):
    """Подготавливает базу знаний из ответов"""
    print("Создаем базу знаний...")
    
    # Объединяем все ответы в одну базу знаний
    all_answers = []
    for item in qa_data:
        answer = item.get('answer', '')
        if answer:
            all_answers.append(answer)
    
    # Создаем чанки из всех ответов
    all_chunks = []
    for answer in all_answers:
        chunks = chunk_text(answer, chunk_size, overlap)
        all_chunks.extend(chunks)
    
    print(f"Создано {len(all_chunks)} текстовых чанков")
    
    # Создаем эмбеддинги для всех чанков
    print("Создаем эмбеддинги...")
    embeddings = []
    for i, chunk in enumerate(all_chunks):
        if i % 10 == 0:
            print(f"Обработано {i}/{len(all_chunks)} чанков")
        
        try:
            embedding_response = create_embeddings(chunk, client=client)
            embeddings.append(embedding_response.data[0])
        except Exception as e:
            print(f"Ошибка при создании эмбеддинга для чанка {i}: {e}")
            continue
    
    print(f"Создано {len(embeddings)} эмбеддингов")
    return all_chunks, embeddings

def semantic_search_wrapper(query, text_chunks, embeddings, k=3):
    """Обертка для семантического поиска"""
    try:
        # Создаем эмбеддинг для запроса
        query_embedding_response = create_embeddings(query, client=client)
        query_embedding = query_embedding_response.data[0].embedding
        
        similarity_scores = []
        
        # Вычисляем схожесть с каждым чанком
        for i, chunk_embedding in enumerate(embeddings):
            # Простое вычисление косинусной схожести
            import numpy as np
            
            vec1 = np.array(query_embedding)
            vec2 = np.array(chunk_embedding.embedding)
            
            # Косинусная схожесть
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarity_scores.append((i, similarity))
        
        # Сортируем по убыванию схожести
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-k наиболее релевантных чанков
        top_indices = [index for index, _ in similarity_scores[:k]]
        return [text_chunks[index] for index in top_indices]
        
    except Exception as e:
        print(f"Ошибка при семантическом поиске: {e}")
        return []

def rag_answer(question, text_chunks, embeddings, max_context_length=1500):
    """Генерирует ответ используя RAG"""
    print(f"\nОбрабатываем вопрос: {question}")
    
    # Находим релевантные чанки
    relevant_chunks = semantic_search_wrapper(question, text_chunks, embeddings, k=3)
    
    if not relevant_chunks:
        print("Не найдено релевантных чанков")
        return "Извините, не могу найти информацию для ответа на ваш вопрос."
    
    # Объединяем релевантные чанки в контекст
    context = "\n\n".join(relevant_chunks)
    
    # Обрезаем контекст если он слишком длинный
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    # Создаем промпт для LLM
    system_prompt = """Вы - помощник службы поддержки интернет магазина. 
Отвечайте на вопросы клиентов на основе предоставленной информации.
Будьте вежливы, информативны и точны.
Если информации недостаточно, честно скажите об этом."""
    
    user_message = f"""Контекст из базы знаний:
{context}

Вопрос клиента: {question}

Ответьте на вопрос клиента, используя информацию из контекста."""
    
    try:
        # Генерируем ответ
        response = generate_response(system_prompt, user_message)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ошибка при генерации ответа: {e}")
        return "Извините, произошла ошибка при генерации ответа."

def main():
    """Основная функция"""
    print("=== RAG СИСТЕМА ДЛЯ ИНТЕРНЕТ МАГАЗИНА ===")
    print("💡 Подсказка: используйте 'python main.py --rebuild' для пересоздания векторной базы")
    
    # Проверяем аргументы командной строки
    rebuild_database = "--rebuild" in sys.argv or "-r" in sys.argv
    
    # Загружаем данные
    qa_data = load_qa_data("ecommerce_qa.json")
    if not qa_data:
        print("Не удалось загрузить данные")
        return
    
    # Проверяем существование векторной базы
    if rebuild_database:
        print("\n🔄 Принудительное пересоздание векторной базы...")
        text_chunks, embeddings = prepare_knowledge_base(qa_data)
        
        if text_chunks and embeddings:
            save_vector_database(text_chunks, embeddings)
        else:
            print("Не удалось подготовить базу знаний")
            return
            
    elif check_vector_database_exists():
        print("\n📦 Найдена существующая векторная база данных")
        text_chunks, embeddings = load_vector_database()
        
        if not text_chunks or not embeddings:
            print("⚠️  Ошибка загрузки, создаем новую базу...")
            text_chunks, embeddings = prepare_knowledge_base(qa_data)
            save_vector_database(text_chunks, embeddings)
    else:
        print("\n🔧 Векторная база не найдена, создаем новую...")
        text_chunks, embeddings = prepare_knowledge_base(qa_data)
        
        if text_chunks and embeddings:
            save_vector_database(text_chunks, embeddings)
        else:
            print("Не удалось подготовить базу знаний")
            return
    
    print(f"\n✅ Векторная база готова к работе!")
    print(f"   📊 Загружено {len(text_chunks)} текстовых чанков")
    print(f"   🔍 Доступно {len(embeddings)} эмбеддингов для поиска")
    
    print("\n=== ТЕСТИРОВАНИЕ RAG СИСТЕМЫ ===")
    
    # Тестируем на нескольких вопросах из JSON
    test_questions = [
        qa_data[0]["question"],  # Первый вопрос
        qa_data[5]["question"],  # Шестой вопрос  
        qa_data[10]["question"], # Одиннадцатый вопрос
        "Какой у вас номер телефона?",  # Новый вопрос
        "Можно ли купить в кредит?"     # Еще один новый вопрос
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- ТЕСТ {i} ---")
        answer = rag_answer(question, text_chunks, embeddings)
        print(f"ВОПРОС: {question}")
        print(f"ОТВЕТ RAG: {answer}")
        print("-" * 80)
    
    # Интерактивный режим
    print("\n=== ИНТЕРАКТИВНЫЙ РЕЖИМ ===")
    print("Введите ваши вопросы (или 'выход' для завершения):")
    
    while True:
        user_question = input("\nВаш вопрос: ").strip()
        
        if user_question.lower() in ['выход', 'exit', 'quit', '']:
            print("До свидания!")
            break
        
        answer = rag_answer(user_question, text_chunks, embeddings)
        print(f"\nОтвет: {answer}")

if __name__ == "__main__":
    main()
