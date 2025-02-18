import scrapy


class ClimateActionTrackerSpider(scrapy.Spider):
    name = "climate_action_tracker"
    allowed_domains = ["climateactiontracker.org"]
    start_urls = ["https://climateactiontracker.org/countries/brazil/"]

    def parse(self, response):
        country_name = response.css('h1::text').get()

        # Return a dictionary of items
        yield {
            'country_name': country_name
    }
