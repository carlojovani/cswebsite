import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'dev-key'

DEBUG = True

ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'users.apps.UsersConfig',
    'faceit_analytics',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'users' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

LANGUAGE_CODE = 'ru'
TIME_ZONE = 'Europe/Berlin'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
LOCAL_DEMOS_ROOT = MEDIA_ROOT / "local_demos"

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'
LOGIN_URL = '/login/'

# Faceit API настройки
FACEIT_API_KEY = '0224f6be-8635-485c-a431-866a1516bd93'
FACEIT_API_URL = 'https://open.faceit.com/data/v4'

# Кастомная модель пользователя
AUTH_USER_MODEL = 'users.CustomUser'

# Celery (Redis broker)
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/0")
CELERY_TASK_ALWAYS_EAGER = os.getenv("CELERY_TASK_ALWAYS_EAGER") == "1"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "UTC"

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": REDIS_URL,
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
    }
}

HEATMAP_OUTPUT_SIZE = 1024
HEATMAP_UPSCALE_FILTER = "LANCZOS"
HEATMAP_BLUR_SIGMA_GRID = 0.8
HEATMAP_BLUR_SIGMA_OUTPUT = 0.0
HEATMAP_NORM_PERCENTILE = 99.0
HEATMAP_GAMMA = 0.55
HEATMAP_ALPHA = 0.95
HEATMAP_BLUR_SIGMA = HEATMAP_BLUR_SIGMA_GRID
HEATMAP_BLUR_RADIUS = HEATMAP_BLUR_SIGMA
HEATMAP_CLIP_PCT = HEATMAP_NORM_PERCENTILE
HEATMAP_TIME_SLICES = [
    (0, 15),
    (15, 30),
    (30, 45),
    (45, 60),
    (60, 90),
    (90, 999),
]
HEATMAP_DEFAULT_SLICE = "0-15"
DEATH_AWARENESS_LOOKBACK_SEC = 5
MULTIKILL_WINDOW_SEC = 10
MULTIKILL_EARLY_THRESHOLD_SEC = 30
