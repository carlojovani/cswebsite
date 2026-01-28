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
celery -A backend.celery worker -l info -P solo
```

## Smoke-check аналитики

1. Откройте профиль игрока и нажмите **Сгенерировать**.
2. В логах Django появится `POST /api/analytics/<profile_id>/start`.
3. Celery воркер получит задачу `task_full_pipeline`.
4. `GET /api/analytics/jobs/<job_id>` будет возвращать переходы `PENDING -> STARTED/PROCESSING -> DONE` и рост `progress` до 100.
5. После `DONE` интерфейс обновится и покажет результаты.

### Настройки Redis

По умолчанию используется `redis://127.0.0.1:6379/0`. Можно переопределить через переменную окружения:

```bash
export REDIS_URL=redis://127.0.0.1:6379/0
```
