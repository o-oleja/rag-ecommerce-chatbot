# Настройка переменных окружения

## Создание .env файла

Для безопасного хранения OpenAI API ключа создайте файл `.env` в корневой папке проекта:

### 1. Создайте файл .env
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Пример содержимого .env файла:
```bash
# OpenAI API ключ
OPENAI_API_KEY=your_openai_api_key_here

# Дополнительные настройки (опционально)
DEBUG=False
LOG_LEVEL=INFO
```

### 3. Безопасность
- **НЕ ДОБАВЛЯЙТЕ** файл `.env` в git repository
- Добавьте `.env` в файл `.gitignore`
- Файл `.env` должен содержать только переменные окружения

### 4. Альтернативный способ (PowerShell)
Вместо .env файла можно установить переменную окружения в PowerShell:
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

## Проверка настройки

Запустите тест конфигурации:
```bash
python -c "from config.config_loader import load_config; config = load_config(); print('API ключ настроен:', 'sk-' in config['openai']['api_key'])"
```

## Использование в коде

Система автоматически загружает переменные из `.env` файла при запуске. Ключ доступен через:
```python
from config.config_loader import load_config

config = load_config()
api_key = config['openai']['api_key']
``` 