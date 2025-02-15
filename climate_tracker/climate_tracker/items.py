"""
Item definitions for the Climate Tracker spider.

Similar to how we used Pydantic models in the ascor-api project, 
these Item classes define the structure and validation for our scraped data.
"""

import scrapy
from scrapy.item import Field


class CountryClimateItem(scrapy.Item):
    """
    Represents climate action data for a country.
    
    Similar to:
    ```python
    class Country(BaseModel):
        country_name: str
        overall_rating: str
        flag_url: str
    ```
    """
    
    # Required fields
    country_name = Field(
        serializer=str,
        required=True
    )
    
    overall_rating = Field(
        serializer=str,
        required=True,
        # Valid ratings from Climate Action Tracker
        choices=[
            'Critically insufficient',
            'Highly insufficient',
            'Insufficient',
            'Almost sufficient',
            'Compatible',
            'Role model'
        ]
    )
    
    flag_url = Field(
        serializer=str,
        required=True
    )
    
    # Optional field - will be populated by pipeline
    flag_path = Field(
        serializer=str,
        required=False
    )
    
    def __repr__(self):
        """String representation of the item."""
        return f"<CountryClimate: {self['country_name']}>"
