"""
Climate Action Tracker Spider

Created by Dr Jon Cardoso-Silva as part of the 
[DS205 course](https://lse-dsi.github.io/DS205) at LSE.

This spider is designed to demonstrate ethical web scraping practices
while collecting climate action data from various countries.

To run the spider for all mapped URLs, use the following command:

    cd climate_tracker
    scrapy crawl climate_action_tracker -O ./data/output.json

To test or run the same spider for a single URL, use the following command:

    cd climate_tracker
    scrapy parse https://climateactiontracker.org/countries/india/ --spider climate_action_tracker -O ./data/output.json

---
Data Usage Notice:
Data and extracted textual content from the Climate Action Tracker website are copyrighted
© 2009-2025 by Climate Analytics and NewClimate Institute. 
All rights reserved.
"""

import scrapy


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
    start_urls = [
        "https://climateactiontracker.org/countries/india/"
    ]

    def parse(self, response):

        """Extract country name, overall rating, flag, and climate indicators."""

 

        # Extract basic country details

        country_name = response.css('h1.headline__title::text').get()

        overall_rating = response.css('.ratings-matrix__overall dd::text').get()

        flag_url = response.css('.headline__flag img::attr(src)').get()

        flag_url = f"https://climateactiontracker.org{flag_url}" if flag_url else None

 

        # Extract climate indicators from the updated structure

        indicators = []

        for indicator in response.css('.ratings-matrix__second-row-cell, .ratings-matrix__third-row-cell'):

            term = indicator.css('dt p::text, dt::text').get()

            term_details = indicator.css('dt i::text').get()

            value = indicator.css('dd b::text').get()

            metric = indicator.css('dd i::text').get()

 

            if term and value:

                indicators.append({

                    "term": term.strip(),

                    "term_details": term_details.strip() if term_details else None,

                    "value": value.strip(),

                    "metric": metric.strip() if metric else None

                })

 

        # Yield the result as a dictionary

        yield {

            'country_name': country_name,

            'overall_rating': overall_rating,

            'flag_url': flag_url,

            'indicators': indicators

        }

 

    def start_requests(self):
        """Initialize the crawl with requests for each start URL.
        
        This method allows for customisation of how the initial requests are made,
        which can be useful for adding headers, cookies, or other request parameters.
        
        Yields:
            scrapy.Request: Request object for each start URL
        """
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse)