services:
  redis:
    image: redis:7-alpine
    container_name: cswebsite-redis
    ports:
      - "6379:6379"
