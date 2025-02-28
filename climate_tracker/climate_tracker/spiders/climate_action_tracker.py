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
    scrapy parse https://climateactiontracker.org/countries/uk/ --spider climate_action_tracker -O ./data/output.json

ACTIVATE   
python -m venv scraping-env
source scraping-env/bin/activate

CRAWL 
cd climate_tracker
scrapy crawl climate_action_tracker -O output.json


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
        "https://climateactiontracker.org/countries/brazil/",
        "https://climateactiontracker.org/countries/china/",
        "https://climateactiontracker.org/countries/usa/",
        "https://climateactiontracker.org/countries/india/",
        "https://climateactiontracker.org/countries/eu/",
        "https://climateactiontracker.org/countries/germany/",
        "https://climateactiontracker.org/countries/australia/",
        "https://climateactiontracker.org/countries/united-kingdom/"
    ]

    def parse(self, response):
        """Extract data from country pages.
        
        Args:
            response (scrapy.http.Response): Response object containing page content
            
        Yields:
            dict: Dictionary containing extracted country data
        """

        country_name = response.css('h1::text').get()
        overall_rating = response.css('.ratings-matrix__overall dd::text').get()        

        # The flag is in a div .headline__flag
        flag_url = response.css('.headline__flag img::attr(src)').get()
        # We need to add the base URL to the flag URL
        flag_url = f"https://climateactiontracker.org{flag_url}"
        
        indicators= list()

        for i, indicator in enumerate(response.xpath("//div[contains(@class, 'ratings-matrix__second-row-cell')]")):
                if i < 3:
                    term = indicator.xpath(".//p/text()[1]").get()  # Extracts "Policies and action"
                    term_details = indicator.xpath(".//p/i/text()").get()
                    value = indicator.xpath(".//dd/b/text()").get()
                    metric = indicator.xpath(".//dd/i/text()").get()
                
                indicators.append({
                    'term': term,
                    'term_details': term_details,
                    'value': value,
                    'metric': metric
                })

                if i == 3:
                    term = indicator.xpath(".//dl/dt/text()").get()
                    value = indicator.xpath(".//dl/dd/b/text()").get()
        
                    indicators.append({
                        'term': term,
                        'value':value
                    })      
        # NET ZERO CURRENTLY WORKING ON
        net_zero_date= response.xpath(".//dl[contains(@class, 'ratings-matrix__net_zero_target')]//dt/text()")
        term_details= response.xpath('.//dl[contains(@class, "ratings-matrix__net_zero_target")]//dd')

        indicators.append({
            'term': net_zero_date,
            't':term_details
        })

        yield {
            'country_name': country_name,
            'overall_rating': overall_rating,
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
            yield scrapy.Request(url=url, callback=self.parse)