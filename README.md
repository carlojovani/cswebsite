# CS Website

## Быстрый старт

1. Установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Запустите Redis локально:

```bash
redis-server
```

Или через Docker:

```bash
docker run --rm -p 6379:6379 redis:7
```

3. Примените миграции и запустите Django:

```bash
python manage.py migrate
python manage.py runserver
```

4. Запустите Celery worker:

```bash
celery -A backend worker -l info
```

### Настройки Redis

По умолчанию используется `redis://127.0.0.1:6379/0`. Можно переопределить через переменную окружения:

```bash
export REDIS_URL=redis://127.0.0.1:6379/0
```
