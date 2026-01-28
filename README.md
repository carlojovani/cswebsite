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
docker compose up -d redis
```

Или напрямую через Docker:

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

## Smoke-check аналитики

1. Откройте профиль игрока и нажмите **Сгенерировать**.
2. В логах Django появится `POST /api/analytics/<profile_id>/start`.
3. Celery воркер получит задачу `task_full_pipeline`.
4. `GET /api/analytics/jobs/<job_id>` будет возвращать переходы `PENDING -> RUNNING -> SUCCESS` и рост `progress` до 100.
5. После `SUCCESS` интерфейс обновится и покажет результаты.

## Демки для расширенной аналитики

По умолчанию демки ищутся в `BASE_DIR/demos/<steam_id>/` (steam_id из PlayerProfile).
Можно использовать папку `BASE_DIR/demos/<faceit_nickname>/`, если так удобнее.
Поддерживаются локальные `.dem` файлы.

### Smoke-check парсинга демо

```bash
python manage.py demo_parse_smoke --profile-id <profile_id> --period last_20
```

### Настройки Redis

По умолчанию используется `redis://127.0.0.1:6379/0`. Можно переопределить через переменные окружения:

```bash
export REDIS_URL=redis://127.0.0.1:6379/0
export CELERY_BROKER_URL=redis://127.0.0.1:6379/0
export CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/0
```

## Production: S3-compatible storage

Для production окружения рекомендуется хранить heatmap PNG и другие медиафайлы в S3-совместимом хранилище
через `django-storages`. Настройте `DEFAULT_FILE_STORAGE` и параметры доступа, чтобы `MEDIA_URL` указывал
на ваш CDN/бакет. Это позволит разгрузить локальный диск и ускорить отдачу изображений.
