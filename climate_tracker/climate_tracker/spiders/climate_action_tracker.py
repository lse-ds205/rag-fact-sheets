import scrapy
import logging
import pandas as pd
from climate_tracker.items import RatingsOverview
from climate_tracker.items import RatingsDescription
from climate_tracker.items import CountryTargets

logger = logging.getLogger(__name__)

class ClimateActionTrackerSpider(scrapy.Spider):
    name = "climate_action_tracker"
    allowed_domains = ["climateactiontracker.org"]
    start_urls = ["https://climateactiontracker.org/countries/"]

    def parse(self, response):
        logger.info('Discovering country URLs ...')
        country_links_xpath = '//a[starts-with(@href, "/countries")]/@href'
        country_links = response.xpath(country_links_xpath).getall()
        logger.info(f'Found {len(country_links)} country links.')

        #Here is where you can put the limit for how many links you want to be going through
        for href in country_links:
            country_url = response.urljoin(href)
            logger.debug(f"Found country URL: {country_url}")
            yield response.follow(
                country_url, 
                callback=self.parse_country
            )

    def parse_country(self, response):

        #This is the collection of the ratings overview item, we are simply collecting the headings, and then how the country rates in each of those categories
        logger.info(f'Starting to parse country page: {response.url}')
        ratings_overview_item = RatingsOverview()
        try:
            ratings_overview_item['country_name'] = self.extract_with_default(response, 'h1::text')
            ratings_overview_item['overall_rating'] = self.extract_with_default(response, 'dt:contains("Overall rating") + dd::text')
            sub_ratings = response.css('div.ratings-matrix__second-row-cell dl > dt:nth-child(1) + dd > b::text').getall()
            ratings_overview_item['policies_action_domestic'] = sub_ratings[0] if len(sub_ratings) > 0 else "NA"
            ratings_overview_item['ndc_target_domestic'] = sub_ratings[1] if len(sub_ratings) > 1 else "NA"
            ratings_overview_item['ndc_target_fair'] = sub_ratings[2] if len(sub_ratings) > 2 else "NA"
            ratings_overview_item['climate_finance'] = sub_ratings[3] if len(sub_ratings) > 3 else "NA"
            ratings_overview_item['net_zero_target_year'] = self.extract_with_default(response, 'div.ratings-matrix__third-row-cell dl.ratings-matrix__net_zero_target dd:nth-child(2) b::text')
            ratings_overview_item['net_zero_target_rating'] = self.extract_with_default(response, 'div.ratings-matrix__third-row-cell dl.ratings-matrix__net_zero_target dd:nth-child(3) b::text', strip=True)
            ratings_overview_item['land_forestry_use'] = self.extract_with_default(response, 'div.ratings-matrix__third-row-cell dl.ratings-matrix__land_use_forestry dd b::text', strip=True)
        except Exception as e:
            logger.error(f'Error parsing country page: {response.url} - {e}')
        yield ratings_overview_item

        #This is the collection of the ratings description item, we are simply collecting the headings, and then the content text for each of those headings    
        ratings_description_item = RatingsDescription()

        try:
            #First we collect all of the containers that have a content block in them
            containers_with_content_block = response.xpath('//div[contains(@class, "container")][.//div[contains(@class, "content-block")]]')

            #Then we iterate over each of those containers and extract the header, rating, and content text, acklowledging that the header may be in a few different places
            for s in containers_with_content_block:
                header = s.css('div.ratings-section-header dt::text').get()
                if header is None:
                    p_text = s.css('div.ratings-section-header p::text').get()
                    if p_text:
                        br_text = s.css('div.ratings-section-header p br + ::text').get()
                        if br_text:
                            header = p_text + " " + br_text
                        else:
                            header = p_text
                rating = None
                #we also make sure to onl collect the rating if it is with the headers that indicate there should be a rating present
                if header is None:
                    header = s.css('h3::text').get().strip()
                    rating = 'no rating'
                else:
                    rating = s.css('div.ratings-section-header dd::text').get()
                content_text = ' '.join(t.strip() for t in s.css('div.content-block ::text').getall() if t.strip())
                
                ratings_description_item['country_name'] = self.extract_with_default(response, 'h1::text')
                ratings_description_item['header'] = header
                ratings_description_item['rating'] = rating
                ratings_description_item['content_text'] = content_text
                yield ratings_description_item

        except Exception as e:
            logger.error(f'Error parsing country texts portion: {response.url} - {e}')

        #Now we can say that the process is over for the scrape on the country page
        logger.info(f'Finished parsing country page: {response.url}')

        #Now we set up the "Target Page Link" to be followed
        target_page_url = response.css('a[href*="/target"]::attr(href)').get()
        if target_page_url:
            target_page_url = response.urljoin(target_page_url)
            yield response.follow(target_page_url, callback=self.parse_country_target)

        logger.info(f'Finished parsing country page: {response.url}')

    def parse_country_target(self, response):
        logger.info(f'Starting to parse country target page: {response.url}')

        country_targets_item = CountryTargets()

        try:
            #First we collect all of the containers that have a content block in them
            containers_with_content_block = response.xpath('//div[contains(@class, "container")][.//div[contains(@class, "content-block")]]')
            for s in containers_with_content_block:
                target = s.css('h3::text').get().strip()
                target_description = ' '.join(t.strip() for t in s.css('div.content-block ::text').getall() if t.strip())
                logger.info(f'Extracted target content: {target} - {target_description}')

                country_targets_item['country_name'] = self.extract_with_default(response, 'h1::text')
                country_targets_item['target'] = target
                country_targets_item['target_description'] = target_description


                yield country_targets_item
            ##Now we need to collect the tables which are in the NDC Target Section Essentially

                

        except Exception as e:
            logger.error(f'Error parsing country target page: {response.url} - {e}')
        # Yield the extracted data as an item or process it further
        # yield TargetItem(data=target_data)

        logger.info(f'Finished parsing country target page: {response.url}')





    #These are functions we use for default and error handling    
    def extract_with_default(self, response, css_selector, default="NA", strip=False):
        value = response.css(css_selector).get()
        if value:
            return value.strip() if strip else value
        else:
            logger.warning(f'Missing value for selector: {css_selector} on page: {response.url}')
            return default

    
    #response.css('#section__overview::text').get().strip()
    #response.css('#section__description-of-cat-ratings *::text').get().strip()




