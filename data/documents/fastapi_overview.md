# FastAPI Overview and Core Architecture

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints. It is built on top of Starlette for the web parts and Pydantic for the data parts.

## Key Features

1. **High Performance**: On par with NodeJS and Go (thanks to Starlette and Uvicorn). One of the fastest Python frameworks available.
2. **Fast to code**: Increases the speed to develop features by about 200% to 300%.
3. **Fewer bugs**: Reduces about 40% of human (developer) induced errors.
4. **Intuitive**: Great editor support with autocompletion everywhere and less time debugging.
5. **Standard-based**: Based on and fully compatible with open standards for APIs: OpenAPI (previously known as Swagger) and JSON Schema.

## Creating a Basic FastAPI Application

To create a basic application in FastAPI, instantiate `FastAPI` and declare endpoint routes using path operation decorators:

```python
from fastapi import FastAPI

app = FastAPI(title="Sample Application", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}
```

## Dependency Injection System

FastAPI has an extremely powerful and intuitive Dependency Injection system. It allows developers to declare dependencies required by route handlers cleanly, such as database sessions, authentication contexts, and service layers.

```python
from fastapi import Depends, FastAPI

app = FastAPI()

def common_parameters(q: str = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

## Async vs Synchronous Route Handlers

FastAPI supports both `async def` and regular `def` endpoint functions:
- If your endpoint calls non-blocking async code (such as `httpx` or an async database driver like `asyncpg`), declare it with `async def`.
- If your endpoint uses blocking synchronous libraries (such as `requests` or synchronous `psycopg2`), declare it with standard `def`. FastAPI runs standard `def` functions in an external threadpool to prevent blocking the event loop.
