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

from datetime import datetime

from climate_tracker.logging import setup_colored_logging

logger = logging.getLogger(__name__)
setup_colored_logging(logger)

def get_indicators(response):
    for indicator in response.css(".ratings-matrix__second-row-cell"):
        term = indicator.css("dt p::text").get()
        # Account for NoneTypes and weird formatting
        if term:
            term = term.strip()
        else:
            term = indicator.css("dt::text").get()
        
        yield {
            "term": term,
            "term_details": indicator.css("i::text").get(),
            "value": indicator.css("b::text").get(),
            "metric": indicator.css("dd i::text").get()
        }

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
        """Extract data from country pages.
        
        Args:
            response (scrapy.http.Response): Response object containing page content
            
        Yields:
            dict: Dictionary containing extracted country data
        
        @url https://climateactiontracker.org/countries/argentina/
        @valid_country
        @valid_indicators 
        @complete_data
        @returns items 1 1
        @url https://climateactiontracker.org/countries/china/
        @valid_country
        @valid_indicators 
        @complete_data
        @returns items 1 1
        @url https://climateactiontracker.org/countries/india/
        @valid_country
        @valid_indicators 
        @complete_data
        @returns items 1 1
        @url https://climateactiontracker.org/countries/eu/
        @valid_country
        @valid_indicators 
        @complete_data
        @returns items 1 1
        @url https://climateactiontracker.org/countries/usa/
        @valid_country
        @valid_indicators 
        @complete_data
        @returns items 1 1
        @url https://climateactiontracker.org/countries/brazil/
        @valid_country
        @valid_indicators 
        @complete_data
        @returns items 1 1
        @url https://climateactiontracker.org/countries/uk/
        @valid_country
        @valid_indicators 
        @complete_data
        @returns items 1 1
        """

        country_name = response.css("h1::text").get()
        overall_rating = response.css(".ratings-matrix__overall dd::text").get()
        indicators = list(get_indicators(response))

        yield {
            "country_name": country_name,
            "overall_rating": overall_rating,
            "indicators": indicators
        }