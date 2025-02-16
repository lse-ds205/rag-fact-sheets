"""
Item pipelines for the Climate Tracker spider.

This module contains pipelines for processing and validating scraped items.
"""

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import logging
from scrapy.exceptions import DropItem
from scrapy.pipelines.files import FilesPipeline
from scrapy import Request

# useful for handling different item types with a single interface
from .items import CountryClimateItem
from .logging import setup_colored_logging


logger = logging.getLogger(__name__)
setup_colored_logging(logger)

class ValidateItemPipeline:

    def process_item(self, item, spider):

        logger.info(f"Validating item: {item}")

        if not isinstance(item, CountryClimateItem):
            raise DropItem(f"Unknown item type: {type(item)}")

        if item['overall_rating'] not in ['Insufficient', 'Compatible']:
            logger.error(f"Dropping item {item} because it has an invalid rating: {item['overall_rating']}")
            raise DropItem(f"Invalid rating: {item['overall_rating']}")
            
        return item

class CountryFlagsPipeline(FilesPipeline):
    """Pipeline for downloading country flag SVG files.
    
    This pipeline:
    1. Downloads flag SVGs from the URLs in items
    2. Stores them with country-specific filenames
    3. Updates items with local file paths
    """
    
    def get_media_requests(self, item, info):
        """Request SVG download if URL is present."""
        if item.get('flag_url'):
            logger.debug(f"Requesting flag download for {item['country_name']}")
            yield Request(
                item['flag_url'],
                meta={'country': item.get('country_name', 'unknown')}
            )

    def file_path(self, request, response=None, info=None):
        """Generate file path for storing the SVG."""
        country = request.meta['country'].lower().replace(' ', '_')
        return f'flags/{country}.svg'

    def item_completed(self, results, item, info):
        """Update item with local file path after download."""
        if results and results[0][0]:  # if success
            item['flag_path'] = results[0][1]['path']
            logger.debug(f"Successfully downloaded flag for {item['country_name']}")
        else:
            logger.warning(f"Failed to download flag for {item['country_name']}")
            item['flag_path'] = None
        return item
