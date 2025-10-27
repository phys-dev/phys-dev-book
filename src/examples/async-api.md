# Async API для кинотеатра

## Структура проекта

```
project/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── main.py
│   ├── api/
│   │   └── v1/
│   │       └── film.py
│   ├── core/
│   │   ├── config.py
│   │   └── logger.py
│   ├── db/
│   │   ├── elastic.py
│   │   └── redis.py
│   ├── models/
│   │   └── film.py
│   └── services/
│       └── film.py
```

## Основные зависимости

```txt
aioredis==1.3.1
elasticsearch[async]==7.9.1
fastapi==0.61.1
orjson==3.4.1
uvicorn==0.12.2
uvloop==0.14.0
```

## Конфигурация

**core/config.py:**

```python
import os
from logging import config as logging_config
from core.logger import LOGGING

logging_config.dictConfig(LOGGING)

PROJECT_NAME = os.getenv('PROJECT_NAME', 'movies')
REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
ELASTIC_HOST = os.getenv('ELASTIC_HOST', '127.0.0.1')
ELASTIC_PORT = int(os.getenv('ELASTIC_PORT', 9200))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

## База данных

**db/elastic.py:**

```python
from typing import Optional
from elasticsearch import AsyncElasticsearch

es: Optional[AsyncElasticsearch] = None

async def get_elastic() -> AsyncElasticsearch:
    return es
```

**db/redis.py:**

```python
from typing import Optional
from aioredis import Redis

redis: Optional[Redis] = None

async def get_redis() -> Redis:
    return redis
```

## Основное приложение

**main.py:**

```python
import logging
import aioredis
import uvicorn
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from api.v1 import film
from core import config
from core.logger import LOGGING
from db import elastic, redis

app = FastAPI(
    title=config.PROJECT_NAME,
    docs_url='/api/openapi',
    openapi_url='/api/openapi.json',
    default_response_class=ORJSONResponse,
)

@app.on_event('startup')
async def startup():
    redis.redis = await aioredis.create_redis_pool(
        (config.REDIS_HOST, config.REDIS_PORT),
        minsize=10,
        maxsize=20
    )
    elastic.es = AsyncElasticsearch(
        hosts=[f'{config.ELASTIC_HOST}:{config.ELASTIC_PORT}']
    )

@app.on_event('shutdown')
async def shutdown():
    await redis.redis.close()
    await elastic.es.close()

app.include_router(film.router, prefix='/api/v1/film', tags=['film'])

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
```

## API слой

**api/v1/film.py:**

```python
from http import HTTPStatus
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from services.film import FilmService, get_film_service

router = APIRouter()

class Film(BaseModel):
    id: str
    title: str

@router.get('/{film_id}', response_model=Film)
async def film_details(
    film_id: str, 
    film_service: FilmService = Depends(get_film_service)
) -> Film:
    film = await film_service.get_by_id(film_id)
    if not film:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, 
            detail='film not found'
        )
    return Film(id=film.id, title=film.title)
```

## Сервисный слой

**services/film.py:**

```python
from functools import lru_cache
from typing import Optional
from aioredis import Redis
from elasticsearch import AsyncElasticsearch
from fastapi import Depends

from db.elastic import get_elastic
from db.redis import get_redis
from models.film import Film

class FilmService:
    def __init__(self, redis: Redis, elastic: AsyncElasticsearch):
        self.redis = redis
        self.elastic = elastic
    
    async def get_by_id(self, film_id: str) -> Optional[Film]:
        film = await self._film_from_cache(film_id)
        if not film:
            film = await self._get_film_from_elastic(film_id)
            if not film:
                return None
            await self._put_film_to_cache(film)
        return film
    
    async def _get_film_from_elastic(self, film_id: str) -> Optional[Film]:
        doc = await self.elastic.get('movies', film_id)
        return Film(**doc['_source'])
    
    async def _film_from_cache(self, film_id: str) -> Optional[Film]:
        data = await self.redis.get(film_id)
        if not data:
            return None
        film = Film.parse_raw(data)
        return film
    
    async def _put_film_to_cache(self, film: Film):
        await self.redis.set(film.id, film.json(), expire=60 * 5)

@lru_cache()
def get_film_service(
    redis: Redis = Depends(get_redis),
    elastic: AsyncElasticsearch = Depends(get_elastic),
) -> FilmService:
    return FilmService(redis, elastic)
```

## Модели

**models/film.py:**

```python
import orjson
from pydantic import BaseModel

def orjson_dumps(v, *, default):
    return orjson.dumps(v, default=default).decode()

class Film(BaseModel):
    id: str
    title: str
    description: str
    
    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps
```
