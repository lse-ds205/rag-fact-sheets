"""
Climate Action Tracker Spider

Created by Dr Jon Cardoso-Silva as part of the 
[DS205 course](https://lse-dsi.github.io/DS205) at LSE.

This spider is designed to demonstrate ethical web scraping practices
while collecting climate action data from various countries.

To run the spider for all mapped URLs, use the following command:

    cd climate_tracker
    scrapy crawl climate_action_tracker -O ./data/output.json

To test the spider using contracts:

    cd climate_tracker
    scrapy check climate_action_tracker

---
Data Usage Notice:
Data and extracted textual content from the Climate Action Tracker website are copyrighted
© 2009-2025 by Climate Analytics and NewClimate Institute. 
All rights reserved.
"""

import scrapy
import logging

from climate_tracker.items import CountryClimateItem
from climate_tracker.logging import setup_colored_logging

logger = logging.getLogger(__name__)
setup_colored_logging(logger)

class ClimateActionTrackerSpider(scrapy.Spider):
    """Spider for scraping country data from Climate Action Tracker website.
    
    This spider collects climate action data for various countries, including their
    names, ratings, and other climate-related metrics.
    
    Attributes:
        name (str): Identifier for the spider used in scrapy crawl commands
        allowed_domains (list): List of domains the spider is restricted to
        start_urls (list): Initial URLs to begin the crawl from
    """
    
    name = "climate_action_tracker"
    allowed_domains = ["climateactiontracker.org"]
    # Instead of listing all URLs, start from the countries page
    start_urls = ["https://climateactiontracker.org/countries/"]

    def parse(self, response):
        """Extract country URLs from the main countries page.
        """
        logger.info("Discovering country URLs...")

        # Find all country links
        country_links_xpath = '//a[starts-with(@href, "/countries")]/@href'
        country_links = response.xpath(country_links_xpath).getall()

        for href in country_links[0:4]:
            country_url = response.urljoin(href)
            logger.debug(f"Found country URL: {country_url}")
            yield response.follow(
                country_url, 
                callback=self.parse_country
            )

    def parse_country(self, response):
        """Extract data from individual country pages.
        
        @url https://climateactiontracker.org/countries/brazil/
        @valid_country
        @returns items 1 1
        @scrapes country_name overall_rating flag_url
        """
        logger.info(f"Parsing country page: {response.url}")
        
        # Create a new item instance
        item = CountryClimateItem()
        
        try:
            # Extract and assign fields with validation
            item['country_name'] = response.css('h1::text').get()
            if not item['country_name']:
                raise ValueError("Country name not found")
                
            item['overall_rating'] = response.css('.ratings-matrix__overall dd::text').get()
            if not item['overall_rating']:
                raise ValueError("Rating not found")
            
            # Extract flag URL with validation
            flag_url = response.css('.headline__flag img::attr(src)').get()
            if flag_url:
                item['flag_url'] = f"https://climateactiontracker.org{flag_url}"
            else:
                logger.warning(f"No flag found for {item['country_name']}")
                item['flag_url'] = None

            logger.debug(f"Successfully parsed data for {item['country_name']}")
            return item
            
        except Exception as e:
            logger.error(f"Error parsing {response.url}: {str(e)}")
            raise