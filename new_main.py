"""
Главный файл RAG системы для интернет-магазина
Связывает все модули вместе для выполнения полного пайплайна
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Добавляем корневую директорию в path для импортов
sys.path.insert(0, str(Path(__file__).parent))

from config.config_loader import load_config
from services.knowledge_processing import KnowledgeProcessor
from services.query_processing import QueryProcessor


def setup_logging(config):
    """Настраивает логирование"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_system.log', encoding='utf-8')
        ]
    )


def run_tests():
    """Запускает тесты системы"""
    import subprocess
    
    print("🧪 Запускаем тесты...")
    
    try:
        result = subprocess.run(['python', '-m', 'pytest', 'tests/', '-v'], 
                              capture_output=True, text=True)
        
        print("Вывод тестов:")
        print(result.stdout)
        
        if result.stderr:
            print("Ошибки:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✅ Все тесты прошли успешно!")
        else:
            print("❌ Некоторые тесты провалились")
            
        return result.returncode == 0
        
    except FileNotFoundError:
        print("❌ pytest не найден. Установите: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Ошибка запуска тестов: {e}")
        return False


def run_interactive_mode(knowledge_processor, query_processor, config):
    """Запускает интерактивный режим для общения с RAG системой"""
    print("\n" + "="*60)
    print("🤖 ИНТЕРАКТИВНЫЙ РЕЖИМ RAG СИСТЕМЫ")
    print("="*60)
    
    # Загружаем или создаем векторную базу
    qa_file_path = config['data']['qa_file_path']
    vector_db = knowledge_processor.get_or_create_vector_database(qa_file_path)
    
    # Валидируем базу
    if not knowledge_processor.validate_vector_database(vector_db):
        print("❌ Векторная база не прошла валидацию")
        return
        
    print("✅ Векторная база готова к работе!")
    print(f"📊 Статистика: {vector_db.get_stats()}")
    print("\nВведите ваши вопросы (или 'выход' для завершения):")
    print("-" * 60)
    
    while True:
        try:
            user_question = input("\n💬 Ваш вопрос: ").strip()
            
            if user_question.lower() in ['выход', 'exit', 'quit', '']:
                print("\n👋 До свидания!")
                break
                
            # Обрабатываем запрос
            result = query_processor.process_query(user_question, vector_db)
            
            print(f"\n🤖 Ответ: {result['answer']}")
            print(f"📈 Оценка качества: {result['evaluation_score']:.1f}/10")
            
            if result.get('relevant_documents'):
                print(f"📚 Найдено релевантных документов: {len(result['relevant_documents'])}")
                
            if not result['success']:
                print(f"⚠️ Ошибка: {result.get('error', 'Неизвестная ошибка')}")
                
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")


def run_batch_test(knowledge_processor, query_processor, config):
    """Запускает пакетное тестирование на примерах из данных"""
    print("\n" + "="*60)
    print("🔬 РЕЖИМ ПАКЕТНОГО ТЕСТИРОВАНИЯ")
    print("="*60)
    
    # Загружаем данные и векторную базу
    qa_file_path = config['data']['qa_file_path']
    qa_data = knowledge_processor.load_qa_data(qa_file_path)
    vector_db = knowledge_processor.get_or_create_vector_database(qa_file_path)
    
    # Берем первые 5 вопросов для тестирования
    test_questions = [item['question'] for item in qa_data[:5]]
    
    print(f"🧪 Тестируем на {len(test_questions)} вопросах...")
    
    # Обрабатываем вопросы пакетом
    results = query_processor.batch_process_queries(test_questions, vector_db)
    
    # Выводим результаты
    print("\nРезультаты тестирования:")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['success'] else "❌"
        score = result['evaluation_score']
        
        print(f"\n{status} ТЕСТ {i}")
        print(f"Вопрос: {result['question']}")
        print(f"Ответ: {result['answer'][:100]}...")
        print(f"Оценка: {score:.1f}/10")
        
    # Выводим общую статистику
    stats = query_processor.get_processing_stats(results)
    print("\n" + "="*60)
    print("📊 ОБЩАЯ СТАТИСТИКА")
    print("="*60)
    print(f"Всего запросов: {stats['total_queries']}")
    print(f"Успешных: {stats['successful_queries']}")
    print(f"Неудачных: {stats['failed_queries']}")
    print(f"Успешность: {stats['success_rate']:.1%}")
    print(f"Средняя оценка: {stats['average_evaluation_score']:.1f}/10")
    print(f"Диапазон оценок: {stats['min_evaluation_score']:.1f} - {stats['max_evaluation_score']:.1f}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='RAG система для интернет-магазина')
    parser.add_argument('--config', '-c', default='config/config.yaml', 
                       help='Путь к файлу конфигурации')
    parser.add_argument('--rebuild', '-r', action='store_true',
                       help='Принудительно пересоздать векторную базу')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Запустить тесты системы')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Запустить пакетное тестирование')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Запустить интерактивный режим')
    
    args = parser.parse_args()
    
    print("🚀 Запуск RAG системы для интернет-магазина")
    print("=" * 50)
    
    try:
        # Загружаем конфигурацию
        print(f"📁 Загружаем конфигурацию из {args.config}")
        config = load_config(args.config)
        
        # Настраиваем логирование
        setup_logging(config)
        logging.info("RAG система запущена")
        
        # Запускаем тесты если нужно
        if args.test:
            success = run_tests()
            if not success:
                print("❌ Тесты провалились, завершаем работу")
                return 1
                
        # Инициализируем процессоры
        print("🔧 Инициализируем процессоры...")
        knowledge_processor = KnowledgeProcessor(config)
        query_processor = QueryProcessor(config)
        
        # Принудительное пересоздание базы
        if args.rebuild:
            print("🔄 Принудительное пересоздание векторной базы...")
            qa_file_path = config['data']['qa_file_path']
            vector_db = knowledge_processor.get_or_create_vector_database(
                qa_file_path, force_rebuild=True
            )
            print("✅ Векторная база пересоздана")
            
        # Выбираем режим работы
        if args.batch:
            run_batch_test(knowledge_processor, query_processor, config)
        elif args.interactive or not any([args.test, args.batch]):
            # По умолчанию запускаем интерактивный режим
            run_interactive_mode(knowledge_processor, query_processor, config)
            
        logging.info("RAG система завершила работу")
        print("\n✅ Работа завершена успешно")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Работа прервана пользователем")
        return 1
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        logging.error(f"Критическая ошибка: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 