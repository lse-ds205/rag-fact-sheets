import scrapy
from climate_tracker.items import ClimateTrackerItem
import logging
import json

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ClimateActionTrackerSpider(scrapy.Spider):
    name = "climate_action_tracker"
    allowed_domains = ["climateactiontracker.org"]
    start_urls = [
        "https://climateactiontracker.org/countries/uk/",
        "https://climateactiontracker.org/countries/brazil/",
        "https://climateactiontracker.org/countries/china/",
        "https://climateactiontracker.org/countries/usa/",
        "https://climateactiontracker.org/countries/india/",
        "https://climateactiontracker.org/countries/eu/",
        "https://climateactiontracker.org/countries/germany/",
        "https://climateactiontracker.org/countries/australia/"
    ]

    def parse(self, response):
        """Extracts country-level data and follows the 'Policies & Action' page."""
        logger.info(f"Scraping country overview from {response.url}")

        item = ClimateTrackerItem()

        # ✅ Extract Country Name
        item["country_name"] = response.css("h1::text").get(default="Unknown").strip()

        # ✅ Extract Overall Rating
        item["overall_rating"] = response.css(".ratings-matrix__overall dd::text").get(default="No rating available").strip()

        # ✅ Extract Flag URL
        item["flag_url"] = response.css(".headline__flag img::attr(src)").get(default="No flag available")

        # ✅ Follow 'Policies & Action' page if available
        policy_page_link = response.xpath("//a[contains(text(), 'Policies & Action')]/@href").get()
        if policy_page_link:
            full_policy_url = response.urljoin(policy_page_link)
            logger.info(f"Following policy page: {full_policy_url}")
            yield scrapy.Request(full_policy_url, callback=self.parse_policy_page, meta={'item': item})
        else:
            logger.warning(f"No Policies & Action page found for {item['country_name']}")
            yield item  # Save item even if there's no policy page

    def parse_policy_page(self, response):
        """Extracts policy data from the 'Policies & Action' page."""
        item = response.meta['item']
        logger.info(f"Scraping policy data from {response.url}")

        # ✅ Extract headings (clean whitespace)
        headings = [h.strip() for h in response.css('div.content-section__left-side > h3::text').getall()]

        # ✅ Extract text blocks
        paragraphs = response.css('div.content-block > p::text').getall()

        # ✅ Match each heading to corresponding text
        sections = []
        text_index = 0

        for heading in headings:
            section_text = []
            
            # Continue collecting paragraphs until a new section is detected
            while text_index < len(paragraphs):
                section_text.append(paragraphs[text_index])
                text_index += 1

                # Stop collecting if the next paragraph seems like a new section (contains a heading)
                if text_index < len(paragraphs) and any(h in paragraphs[text_index] for h in headings):
                    break

            # Store in a structured dictionary
            sections.append({"title": heading, "content": " ".join(section_text)})

        # ✅ Store sections in the Scrapy item
        item["policy_sections"] = sections

        # ✅ Log the extracted structured data
        logger.info(json.dumps(sections, indent=4))

        yield item  # Save the item with extracted policy data
