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
import requests

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
    
    This spider collects climate action data for all tracked countries, including their
    names, ratings, and other climate-related metrics.
    
    Attributes:
        name (str): Identifier for the spider used in scrapy crawl commands
        allowed_domains (list): List of domains the spider is restricted to
        start_urls (list): Initial URLs to begin the crawl from
    """
    
    name = "climate_action_tracker"
    allowed_domains = ["climateactiontracker.org"]

    countries_response = requests.get("https://climateactiontracker.org/countries/")
    countries_response.raise_for_status()
    start_urls = [
        "https://climateactiontracker.org" + country_path for country_path in
        scrapy.Selector(countries_response).css(".countries-menu-dropdown .hidden-xs a::attr(href)").getall()
    ]

    def parse(self, response):
        """Extract data from country pages.
        
        Args:
            response (scrapy.http.Response): Response object containing page content
            
        Yields:
            dict: Dictionary containing extracted country data
        """
        country_name = response.css("h1::text").get()
        overall_rating = response.css(".ratings-matrix__overall dd::text").get()
        indicators = list(get_indicators(response))

        yield {
            "country_name": country_name,
            "overall_rating": overall_rating,
            "indicators": indicators
        }

    def start_requests(self):
        """Initialize the crawl with requests for each start URL.
        
        This method allows for customisation of how the initial requests are made,
        which can be useful for adding headers, cookies, or other request parameters.
        
        Yields:
            scrapy.Request: Request object for each start URL
        """
        for url in self.start_urls:
            yield scrapy.Request(url = url, callback = self.parse)