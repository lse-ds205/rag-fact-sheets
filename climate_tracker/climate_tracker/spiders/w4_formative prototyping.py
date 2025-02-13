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
        "https://climateactiontracker.org/countries/united-states/",
        "https://climateactiontracker.org/countries/india/",
        "https://climateactiontracker.org/countries/france/",
        "https://climateactiontracker.org/countries/germany/",
        "https://climateactiontracker.org/countries/australia/",
        "https://climateactiontracker.org/countries/united-kingdom/"
    ]

    def start_requests(self):
        """Initialize the crawl with requests for each start URL.
            
        This method allows for customisation of how the initial requests are made,
        which can be useful for adding headers, cookies, or other request parameters.
            
        Yields:
        scrapy.Request: Request object for each start URL
        """
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        """Extract data from country pages.
        
        Args:
            response (scrapy.http.Response): Response object containing page content
            
        Yields:
            dict: Dictionary containing extracted country data
        """
        country_name = response.css('h1::text').get()
        overall_rating = response.css('.ratings-matrix__overall dd::text').get()

        indicator_classes = [
            'ratings-matrix__second-row-cell ratings-matrix__tile--ORANGE', 
            'ratings-matrix__second-row-cell ratings-matrix__tile--RED', 
            'ratings-matrix__second-row-cell ratings-matrix__tile--GRAY'
            ]

        indicator_names = [
            "policies_and_action", "conditional_ndc_target", "unconditional_ndc_target"
            ]


        for indicator in indicator_classes and indicator_name in indicator_names: 
            term = response.css('.'+ indicator + ' p::text').get()
            term_details = response.css('.'+ indicator +' p i::text').get()
            value = response.css('.'+ indicator +' dd b::text').get()
            metric = response.css('.' + indicator ' dd i::text').get()


        #there are TWO instances of organe class so have to manually get second orange class

        yield {
            'country_name': country_name,
            'overall_rating': overall_rating,
    
            "indicators": [{
            "term": term,
            "term_details": term_details,
            "value": value,
            "metric": metric
            },
            {
            "term": "Conditional NDC target",
            "term_details": "against modelled domestic pathways",
            "value": "Highly insufficient",
            "metric": "< 4°C World"
            },
           # // ... other indicators
            ]
        }

"""
Target output:
[{
  "country_name": "India",
  "overall_rating": "Highly insufficient",
  "indicators": [{
      "term": "Policies and action",
      "term_details": "against fair share",
      "value": "Insufficient",
      "metric": "< 3°C World"
    },
    {
      "term": "Conditional NDC target",
      "term_details": "against modelled domestic pathways",
      "value": "Highly insufficient",
      "metric": "< 4°C World"
    },
    // ... other indicators
  ]
}]
"""
