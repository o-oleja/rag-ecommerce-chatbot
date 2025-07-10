"""
Загрузчик конфигурации для RAG системы
Поддерживает переменные окружения и валидацию настроек
"""

import os
import yaml
import logging
from typing import Dict, Any
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class ConfigLoader:
    """Класс для загрузки и валидации конфигурации"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = None
        
    def load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из YAML файла с обработкой переменных окружения"""
        # Загружаем переменные из .env файла, если он существует
        self._load_dotenv()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_text = file.read()
                
            # Заменяем переменные окружения
            config_text = self._substitute_env_variables(config_text)
            
            # Парсим YAML
            self.config = yaml.safe_load(config_text)
            
            # Валидируем конфигурацию
            self._validate_config()
            
            logging.info(f"Конфигурация успешно загружена из {self.config_path}")
            return self.config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл конфигурации не найден: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Ошибка парсинга YAML: {e}")
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки конфигурации: {e}")
    
    def _load_dotenv(self) -> None:
        """Загружает переменные окружения из .env файла, если он существует"""
        if DOTENV_AVAILABLE:
            env_path = Path(".env")
            if env_path.exists():
                load_dotenv(env_path)
                logging.info("Переменные окружения загружены из .env файла")
            else:
                logging.debug(".env файл не найден, используются системные переменные окружения")
        else:
            logging.warning("python-dotenv не установлен. Для автоматической загрузки .env файлов установите: pip install python-dotenv")
            
    def _substitute_env_variables(self, text: str) -> str:
        """Заменяет переменные окружения в формате ${VAR_NAME}"""
        import re
        
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))  # Возвращаем исходное значение если переменная не найдена
            
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, text)
        
    def _validate_config(self) -> None:
        """Валидирует обязательные поля конфигурации"""
        required_sections = ['knowledge_processing', 'query_processing', 'openai', 'data']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Отсутствует обязательная секция в конфигурации: {section}")
                
        # Проверяем, что модели эмбеддингов совпадают
        kp_model = self.config['knowledge_processing']['embedding_model']
        qp_model = self.config['query_processing']['embedding_model']
        
        if kp_model != qp_model:
            raise ValueError(
                f"Модели эмбеддингов должны совпадать: "
                f"knowledge_processing={kp_model}, query_processing={qp_model}"
            )
            
        # Проверяем наличие API ключа
        api_key = self.config['openai']['api_key']
        if not api_key or api_key.startswith('${'):
            logging.warning("OpenAI API ключ не установлен или не найден в переменных окружения")
            
    def get(self, key_path: str, default=None) -> Any:
        """Получает значение из конфигурации по пути (например: 'openai.api_key')"""
        if self.config is None:
            raise RuntimeError("Конфигурация не загружена. Вызовите load_config() сначала.")
            
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Удобная функция для загрузки конфигурации"""
    loader = ConfigLoader(config_path)
    return loader.load_config()


def get_config_value(config: Dict[str, Any], key_path: str, default=None) -> Any:
    """Получает значение из словаря конфигурации по пути"""
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default 