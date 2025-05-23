from typing import List, Optional, TypedDict, Union
from PIL import Image
import copy
from pydantic import BaseModel, Field
import numpy as np


def default_image() -> np.ndarray:
    """Returns a default blank image."""
    # return Image.new("RGB", (224, 224))
    return np.zeros((224, 224, 3), dtype=np.uint8)


class FashionItem(BaseModel):
    item_id: Optional[int] = Field(
        default=None,
        description="Unique ID of the item, mapped to `id` in the ItemLoader"
    )
    category: Optional[str] = Field(
        default="",
        description="Category of the item"
    )
    image: Optional[np.ndarray] = Field(
        default_factory=default_image,
        description="Image of the item"
    )
    description: Optional[str] = Field(
        default="",
        description="Description of the item"
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional metadata for the item"
    )
    embedding: Optional[np.ndarray] = Field(
        default=None,
        description="Embedding of the item"
    )

    class Config:
        arbitrary_types_allowed = True

    
class FashionCompatibilityQuery(BaseModel):
    outfit: List[FashionItem] = Field(
        default_factory=list,
        description="List of fashion items"
    )

class FashionComplementaryQuery(BaseModel):
    outfit: List[FashionItem] = Field(
        default_factory=list,
        description="List of fashion items"
    )
    category: str = Field(
        default="",
        description="Category of the target outfit"
    )
    
    
class FashionCompatibilityData(TypedDict):
    label: Union[
        int, 
        List[int]
    ]
    query: Union[
        FashionCompatibilityQuery, 
        List[FashionCompatibilityQuery]
    ]
    
    
class FashionFillInTheBlankData(TypedDict):
    query: Union[
        FashionComplementaryQuery,
        List[FashionComplementaryQuery]
    ]
    label: Union[
        int,
        List[int]
    ]
    answers: Union[
        List[FashionItem],
        List[List[FashionItem]]
    ]
    
    
class FashionTripletData(TypedDict):
    query: Union[
        FashionComplementaryQuery,
        List[FashionComplementaryQuery]
    ]
    answer: Union[
        FashionItem,
        List[FashionItem]
    ]