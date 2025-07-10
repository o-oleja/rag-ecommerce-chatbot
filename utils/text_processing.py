"""
Утилиты для обработки текста в RAG системе
"""

from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Разбивает текст на части (чанки) с перекрытием
    
    Args:
        text (str): Исходный текст для разбивки
        chunk_size (int): Размер каждого чанка в символах
        overlap (int): Количество символов перекрытия между чанками
        
    Returns:
        List[str]: Список текстовых чанков
        
    Raises:
        ValueError: Если overlap >= chunk_size
    """
    if overlap >= chunk_size:
        raise ValueError("Перекрытие не может быть больше или равно размеру чанка")
        
    if not text or not text.strip():
        return []
        
    chunks = []
    text = text.strip()
    
    # Если текст короче chunk_size, возвращаем его целиком
    if len(text) <= chunk_size:
        chunks.append(text)
        return chunks
    
    # Разбиваем текст на чанки с перекрытием
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        # Если это не последний чанк и конец попадает на середину слова,
        # пытаемся найти ближайший пробел или знак препинания
        if end < len(text):
            # Ищем ближайший пробел в последних 50 символах чанка
            search_start = max(end - 50, start)
            last_space = text.rfind(' ', search_start, end)
            
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:  # Добавляем только непустые чанки
            chunks.append(chunk)
            
        # Вычисляем начальную позицию для следующего чанка
        start = end - overlap
        
        # Если следующий чанк будет полностью перекрываться с текущим, прерываем
        if start >= len(text):
            break
    
    return chunks


def clean_text(text: str) -> str:
    """
    Очищает текст от лишних символов и нормализует пробелы
    
    Args:
        text (str): Исходный текст
        
    Returns:
        str: Очищенный текст
    """
    if not text:
        return ""
        
    # Убираем лишние пробелы и переносы строк
    import re
    
    # Заменяем множественные пробелы и переносы на одинарные
    text = re.sub(r'\s+', ' ', text)
    
    # Убираем пробелы в начале и конце
    text = text.strip()
    
    return text


def extract_qa_content(qa_item: dict) -> str:
    """
    Извлекает контент из элемента вопрос-ответ
    
    Args:
        qa_item (dict): Элемент с ключами 'question' и 'answer'
        
    Returns:
        str: Объединенный контент вопроса и ответа
    """
    question = qa_item.get('question', '').strip()
    answer = qa_item.get('answer', '').strip()
    
    # Объединяем вопрос и ответ для лучшего контекста при поиске
    if question and answer:
        return f"Вопрос: {question}\nОтвет: {answer}"
    elif answer:
        return answer
    elif question:
        return question
    else:
        return ""


def prepare_text_for_embedding(text: str) -> str:
    """
    Подготавливает текст для создания эмбеддинга
    
    Args:
        text (str): Исходный текст
        
    Returns:
        str: Подготовленный текст
    """
    # Очищаем текст
    text = clean_text(text)
    
    # Ограничиваем длину текста для эмбеддинга (OpenAI имеет лимиты)
    max_length = 8000  # Безопасная длина для text-embedding-3-small
    
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0]  # Обрезаем по последнему пробелу
        
    return text 