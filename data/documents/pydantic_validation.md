# Pydantic V2 Data Validation and Serialization

Pydantic is the most widely used data validation library for Python. In Pydantic V2, the core validation logic has been rewritten in Rust (`pydantic-core`), resulting in a 5x to 50x performance increase over V1.

## Defining Models with BaseModel

Data models in Pydantic are defined by inheriting from `pydantic.BaseModel`. Fields are annotated with Python standard types:

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class UserProfile(BaseModel):
    id: int
    username: str = Field(min_length=3, max_length=50)
    email: str
    is_active: bool = True
    signup_date: Optional[datetime] = None
```

## Custom Field Validation with @field_validator

Pydantic V2 provides `@field_validator` to implement custom validation logic on specific attributes before or after standard validation occurs.

```python
from pydantic import BaseModel, field_validator

class UserRegistration(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def validate_username_alphanumeric(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric without spaces")
        return v.lower()
```

## Serialization and Exporting

Pydantic V2 separates validation from serialization. You can serialize models into dictionaries or JSON strings using dedicated methods:
- `model.model_dump()`: Converts model instances into standard Python dictionaries.
- `model.model_dump_json()`: Serializes model instances directly into JSON strings.
- `model.model_validate(raw_data)`: Validates raw dictionary data into a typed model instance.
- `model.model_validate_json(json_str)`: Parses and validates a JSON string directly without intermediate python dictionary allocation.

