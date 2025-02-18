"""
Item pipelines for the Climate Tracker spider.

This module contains pipelines for processing and validating scraped items.
"""

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import json
import logging
import pycountry

from datetime import datetime

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

        logger.debug(f"Validating item: {item}")

        for field in ['country_name', 'overall_rating', 'flag_url']:
            if not item.get(field):
                logger.warn(f"Dropping item {item} because it has no {field}")
                raise DropItem(f"Invalid {field}: {item[field]}")

        # TODO: Test how robust this country search actually is
        search_country = pycountry.countries.get(name=item['country_name'])
        if not search_country:
            logger.warn(f"Dropping item {item} because it has an invalid country name: {item['country_name']}")
            raise DropItem(f"Invalid country name: {item['country_name']}")
        else:
            logger.debug(f"Found country: {search_country}")
            
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
            yield Request(item['flag_url'])

    def file_path(self, request, response=None, info=None, *, item=None):
        """Generate file path for storing the SVG."""
        country = item['country_name'].lower().replace(' ', '_')
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