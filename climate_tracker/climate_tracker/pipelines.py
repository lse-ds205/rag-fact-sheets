"""
Item pipelines for the Climate Tracker spider.

This module contains pipelines for processing and validating scraped items.
"""

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import logging

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
from .logging import setup_colored_logging


logger = logging.getLogger(__name__)
setup_colored_logging(logger)

class ValidateItemPipeline:
    """Pipeline for validating scraped items.
    
    This pipeline ensures that:
    1. Items are of the correct type
    2. Required fields are present
    3. Field values are valid
    """
    
    def open_spider(self, spider):
        """Called when spider starts."""
        logger.info("ValidateItemPipeline started")
    
    def close_spider(self, spider):
        """Called when spider ends."""
        logger.info("ValidateItemPipeline finished")
    
    def process_item(self, item, spider):
        """Process and validate each scraped item.
        
        Args:
            item: The scraped item to validate
            spider: The spider that scraped the item
            
        Returns:
            item: The validated item
            
        Raises:
            DropItem: If the item fails validation
        """
        logger.debug(f"Processing item: {item['country_name']}")
        
        adapter = ItemAdapter(item)
        
        # Check required fields
        required_fields = ['country_name', 'overall_rating', 'flag_url']
        for field in required_fields:
            if not adapter.get(field):
                raise DropItem(f"Missing {field}")
        
        # Validate rating values
        valid_ratings = [
            'Critically insufficient',
            'Highly insufficient',
            'Insufficient',
            'Almost sufficient',
            'Compatible',
            'Role model'
        ]
        
        if adapter['overall_rating'] not in valid_ratings:
            raise DropItem(
                f"Invalid rating: {adapter['overall_rating']}"
            )
        
        # Validate flag URL format
        if not adapter['flag_url'].startswith('https://'):
            raise DropItem(
                f"Invalid flag URL: {adapter['flag_url']}"
            )
        
        return item


class ClimateTrackerPipeline:
    """Legacy pipeline - kept for reference."""
    def process_item(self, item, spider):
        return item
