from typing import Optional

from pydantic import BaseModel, Field, model_validator


class EstimateRequest(BaseModel):
    """The customer input used to produce a jewelry estimate."""

    brief: str = Field(default="", description="Optional description of the jewelry item")
    image: Optional[str] = Field(
        default=None,
        description="Optional image URL or base64-encoded image data",
    )

    @model_validator(mode="after")
    def require_brief_or_image(self):
        if not self.brief.strip() and not self.image:
            raise ValueError("Provide a brief, an image, or both")
        return self


class ExtractedSpecifications(BaseModel):
    """Normalized jewelry characteristics extracted from an item brief."""

    item_type: str = Field(description="For example: ring, chain, necklace, or bracelet")
    metal: str = Field(description="For example: gold, silver, platinum, or unknown")
    purity: str = Field(description="For example: 18K, 22K, 925, 950, or unknown")
    gemstone: str = Field(description="Gemstone name, none, or unknown")
    target_weight_grams: float = Field(
        ge=0,
        description="Estimated item weight in grams; use 0 when unknown",
    )
