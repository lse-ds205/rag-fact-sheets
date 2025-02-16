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
from .items import CountryClimateItem
from scrapy.exceptions import DropItem
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
