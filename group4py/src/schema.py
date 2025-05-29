from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Tuple

class ChunkModel(BaseModel):
    placeholder: Any = Field(default=None, description="A placeholder for any type of data")

class EmbeddingModel(BaseModel):
    placeholder: Any = Field(default=None, description="A placeholder for any type of data")