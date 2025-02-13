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

        indicators_list = list()
        # now for each indicator square in the matrix
        for idx, indicator in enumerate(response.css('div.ratings-matrix__second-row-cell')):

            # for all indicators which aren't climate finance (the 4th square)
            if idx < 3:
                term = indicator.css('div.ratings-matrix__second-row-cell')[0].css('dl p::text').get()
                term_details = indicator.css('div.ratings-matrix__second-row-cell')[0].css('dl i::text').get()
                value = indicator.css('div.ratings-matrix__second-row-cell')[0].css('dd b::text').get()
                metric =  indicator.css('div.ratings-matrix__second-row-cell')[0].css('dd i::text').get()
                
                indicators_list.append({
                    'term': term,
                    'term_details': term_details,
                    'value': value,
                    'metric': metric
                })

            # for the climate finance indicator
            elif idx == 3: 

                term = indicator.css('dl dt::text').get()
                value = indicator.css('dl dd b::text').get()
                
                indicators_list.append({
                    'term': term,
                    'value':value
                })
        
        # for the net zero target indicator:
        net_zero_term = response.css('dl.ratings-matrix__net_zero_target dt::text').get()
        net_zero_term_details = ' '.join([element.strip() for element in response.css('dl.ratings-matrix__net_zero_target dd')[0].css('*::text').getall() if element.strip()])
        net_zero_value =   response.css('dl.ratings-matrix__net_zero_target dd')[0].css('*::text').get().strip()
        indicators_list.append({
            'term': net_zero_term,
            'term_details':net_zero_term_details,
            'value':net_zero_value
        })

        # for the land use and forestry indicator
        land_term = response.css('dl.ratings-matrix__land_use_forestry dt::text').get()
        land_value =  response.css('dl.ratings-matrix__land_use_forestry dd b::text').get().strip()

        indicators_list.append({
            'term': land_term,
            'value':land_value
        })

        yield {
            'country_name': country_name,
            'overall_rating': overall_rating,
            'flag_url': flag_url, 
            'indicators': indicators_list
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