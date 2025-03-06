import scrapy
import logging
import pandas as pd
from climate_tracker.items import RatingsOverview, RatingsDescription, CountryTargets, PolicyAction, NetZeroTargets, Assumptions, CountryDataFiles

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

        # Here is where you can put the limit for how many links you want to be going through
        #This collects all of the names of the countries which are available on the climate action tracker website.
        for href in country_links:  
            country_url = response.urljoin(href)
            logger.debug(f"Found country URL: {country_url}")
            yield response.follow(
                country_url, 
                callback=self.parse_country
            )
    #This is the Parse Function for the Overview Page of the Country
    def parse_country(self, response):
        logger.info(f'Starting to parse country page: {response.url}')

        #First - Collection of how the Country is Rated on Key Indicators
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

        #Second - Collection of the Text Descriptions for the Ratings
        ratings_description_item = RatingsDescription()
        try:
            containers_with_content_block = response.xpath('//div[contains(@class, "container")][.//div[contains(@class, "content-block")]]')
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

        #Third - Collection of the Data Files which provides a visual depiction of the country's climate action status and of the data that makes this graph
        file_links = [
        response.xpath('//div[@data-component-graph-embed]/@data-props-graph-image-url').get(),
        response.xpath('//div[@data-component-graph-embed]/@data-props-scenario-data-url').get()
        ]

        country_data_files_item = CountryDataFiles()
        country_data_files_item['country_name'] = self.extract_with_default(response, 'h1::text')

        for file_url in file_links:
            if file_url:
                full_url = response.urljoin(file_url)
                logger.info(f'Downloading file from: {full_url}')
                yield scrapy.Request(full_url, callback=self.save_file, meta={'country_name': country_data_files_item['country_name'], 'item': country_data_files_item})
        
        
        logger.info(f'Finished parsing country page: {response.url}')

        ##Below are the links to the other pages that are linked to the country page - we will access them via different parse functions

        # Follow the target page link
        target_page_url = response.css('a[href*="/target"]::attr(href)').get()
        if target_page_url:
            target_page_url = response.urljoin(target_page_url)
            yield response.follow(target_page_url, callback=self.parse_country_target)

        # Follow the policies action page link
        policies_action_url = response.css('a[href*="/policies-action"]::attr(href)').get()
        if policies_action_url:
            policies_action_url = response.urljoin(policies_action_url)
            yield response.follow(policies_action_url, callback=self.parse_policies_action)

        # Follow the net-zero targets page link
        net_zero_targets_url = response.css('a[href*="/net-zero-targets"]::attr(href)').get()
        if net_zero_targets_url:
            net_zero_targets_url = response.urljoin(net_zero_targets_url)
            yield response.follow(net_zero_targets_url, callback=self.parse_net_zero_targets)

        #Follow Through to the Assumptions Page
        assumptions_url = response.css('a[href*="/assumptions"]::attr(href)').get()
        if assumptions_url:
            assumptions_url = response.urljoin(assumptions_url)
            yield response.follow(assumptions_url, callback=self.parse_assumptions)

        logger.info(f'Finished parsing country page: {response.url}')


        logger.info(f'Finished parsing country page: {response.url}')
    ###########        
    #This is the function that allows us to store the xlsx file and png file




    #Country Target Page Core
    def parse_country_target(self, response):
        logger.info(f'Starting to parse country target page: {response.url}')

        country_targets_item = CountryTargets()

        try:
            containers_with_content_block = response.xpath('//div[contains(@class, "container")][.//div[contains(@class, "content-block")]]')
            for s in containers_with_content_block:
                target = s.css('h3::text').get().strip()
                target_description = ' '.join(t.strip() for t in s.css('div.content-block ::text').getall() if t.strip())
                logger.info(f'Extracted target content: {target} - {target_description}')

                country_targets_item['country_name'] = self.extract_with_default(response, 'h1::text')
                country_targets_item['target'] = target
                country_targets_item['target_description'] = target_description

                # Extract NDC tables
                table_data = self.parse_tables(s)
                country_targets_item['table_data'] = table_data

                images = s.xpath('.//img/@src').getall()
                country_targets_item['images'] = []
                for image_url in images:
                    if image_url:
                        full_url = response.urljoin(image_url)
                        logger.info(f'Downloading image from: {full_url}')
                        yield scrapy.Request(full_url, callback=self.save_image, meta={'country_name': country_targets_item['country_name'], 'item': country_targets_item})

                yield country_targets_item

        except Exception as e:
            logger.error(f'Error parsing country target page: {response.url} - {e}')

        logger.info(f'Finished parsing country target page: {response.url}')

    def parse_tables(self, container):
        # Store the results
        table_data = {}
        
        # Find all styled tables
        tables = container.xpath('.//div[contains(@class, "styled-table--style")]')
        
        for table in tables:
            # Get table title/header (this could be "unconditional NDC target" or similar)
            table_title = table.xpath('.//th[contains(@class, "htTop") and contains(text(), "target")]//text()').get()
            
            if not table_title:
                # Try alternate ways to get the title
                table_title = table.xpath('.//tr[1]//td//text()[contains(., "target")]').get()
            
            # Clean up the title if needed
            if table_title:
                table_title = table_title.strip()
            else:
                # Use a default or try to identify from context
                table_title = "NDC Target Section"
            
            # Create dict for this table
            table_data[table_title] = {}
            
            # Find all rows with sub-headings and their corresponding values
            rows = table.xpath('.//tr')
            
            for row in rows:
                # First cell is usually the subheading
                subheading = row.xpath('.//td[1]//text()').getall()
                # Second cell is usually the value
                value = row.xpath('.//td[2]//text()').getall()
                
                # Clean and join text elements
                if subheading:
                    subheading = ' '.join([s.strip() for s in subheading if s.strip()])
                    if value:
                        value = ' '.join([v.strip() for v in value if v.strip()])
                        # Store in our dictionary
                        if subheading and value:
                            table_data[table_title][subheading] = value
        
        # For special cases like "2030 unconditional NDC target" section
        unconditional = container.xpath('//div[contains(text(), "unconditional NDC target")]')
        if unconditional:
            table_section = unconditional.xpath('ancestor::div[contains(@class, "styled-table")]')
            section_title = "2030 unconditional NDC target"
            
            table_data[section_title] = {}
            
            # Extract specific fields
            formulation = table_section.xpath('.//td[contains(text(), "Formulation of target")]/following-sibling::td//text()').getall()
            if formulation:
                table_data[section_title]['Formulation of target in NDC'] = ' '.join([f.strip() for f in formulation if f.strip()])
            
            absolute_emissions = table_section.xpath('.//td[contains(text(), "Absolute emissions")]/following-sibling::td//text()').getall()
            if absolute_emissions:
                table_data[section_title]['Absolute emissions level in 2030'] = ' '.join([a.strip() for a in absolute_emissions if a.strip()])
            
            status = table_section.xpath('.//td[contains(text(), "Status")]/following-sibling::td//text()').getall()
            if status:
                table_data[section_title]['Status'] = ' '.join([s.strip() for s in status if s.strip()])
        
        # Net zero section (conditional targets)
        net_zero = container.xpath('//td[contains(text(), "Net zero")]')
        if net_zero:
            table_section = net_zero.xpath('ancestor::div[contains(@class, "styled-table")]')
            section_title = "Net zero & other long term targets"
            
            table_data[section_title] = {}
            
            # Extract specific fields
            formulation = table_section.xpath('.//td[contains(text(), "Formulation of target")]/following-sibling::td//text()').getall()
            if formulation:
                table_data[section_title]['Formulation of target'] = ' '.join([f.strip() for f in formulation if f.strip()])
            
            absolute_emissions = table_section.xpath('.//td[contains(text(), "Absolute emissions")]/following-sibling::td//text()').getall()
            if absolute_emissions:
                table_data[section_title]['Absolute emissions level in 2050'] = ' '.join([a.strip() for a in absolute_emissions if a.strip()])
            
            status = table_section.xpath('.//td[contains(text(), "Status")]/following-sibling::td//text()').getall()
            if status:
                table_data[section_title]['Status'] = ' '.join([s.strip() for s in status if s.strip()])
        
        return table_data

    #Policies and Action Page Core
    def parse_policies_action(self, response):
        logger.info(f'Starting to parse policies action page: {response.url}')

        policy_action_item = PolicyAction()

        try:
            containers_with_content_block = response.xpath('//div[contains(@class, "container")][.//div[contains(@class, "content-block")]]')
            for s in containers_with_content_block:
                policy = s.css('h3::text').get()
                if policy:
                    policy = policy.strip()
                else:
                    policy = s.css('p::text').get().strip()
                action_description = ' '.join(t.strip() for t in s.css('div.content-block ::text').getall() if t.strip())
                logger.info(f'Extracted policy and action content: {policy} - {action_description}')

                policy_action_item['country_name'] = self.extract_with_default(response, 'h1::text')
                policy_action_item['policy'] = policy
                policy_action_item['action_description'] = action_description
                
                #Get any potential images
                images = s.xpath('.//img/@src').getall()
                policy_action_item['images'] = []
                for image_url in images:
                    if image_url:
                        full_url = response.urljoin(image_url)
                        logger.info(f'Downloading image from: {full_url}')
                        yield scrapy.Request(full_url, callback=self.save_image, meta={'country_name': policy_action_item['country_name'], 'item': policy_action_item})

                yield policy_action_item
                
                

        except Exception as e:
            logger.error(f'Error parsing policies action page: {response.url} - {e}')

        logger.info(f'Finished parsing policies action page: {response.url}')

    #Net Zero Targets Page Core
    def parse_net_zero_targets(self, response):
        logger.info(f'Starting to parse net-zero targets page: {response.url}')

        net_zero_targets_item = NetZeroTargets()

        try:
            containers_with_content_block = response.xpath('//div[contains(@class, "container")][.//div[contains(@class, "content-block")]]')
            for s in containers_with_content_block:
                target = s.css('h3::text').get().strip()
                target_description = ' '.join(t.strip() for t in s.css('div.content-block ::text').getall() if t.strip())
                logger.info(f'Extracted net-zero target content: {target} - {target_description}')

                net_zero_targets_item['country_name'] = self.extract_with_default(response, 'h1::text')
                net_zero_targets_item['target'] = target
                net_zero_targets_item['target_description'] = target_description

                # Extract images
                images = s.xpath('.//img/@src').getall()
                net_zero_targets_item['images'] = []
                for image_url in images:
                    if image_url:
                        full_url = response.urljoin(image_url)
                        logger.info(f'Downloading image from: {full_url}')
                        yield scrapy.Request(full_url, callback=self.save_image, meta={'country_name': net_zero_targets_item['country_name'], 'item': net_zero_targets_item})

                yield net_zero_targets_item

        except Exception as e:
            logger.error(f'Error parsing net-zero targets page: {response.url} - {e}')

        logger.info(f'Finished parsing net-zero targets page: {response.url}')

    #Assumptions Core
    def parse_assumptions(self, response):
        logger.info(f'Starting to parse assumptions page: {response.url}')

        assumptions_item = Assumptions()

        try:
            containers_with_content_block = response.xpath('//div[contains(@class, "container")][.//div[contains(@class, "content-block")]]')
            for s in containers_with_content_block:
                assumption = s.css('h3::text').get().strip()
                assumption_description = ' '.join(t.strip() for t in s.css('div.content-block ::text').getall() if t.strip())
                logger.info(f'Extracted assumption content: {assumption} - {assumption_description}')

                assumptions_item['country_name'] = self.extract_with_default(response, 'h1::text')
                assumptions_item['assumption'] = assumption
                assumptions_item['assumption_description'] = assumption_description

                yield assumptions_item

        except Exception as e:
            logger.error(f'Error parsing assumptions page: {response.url} - {e}')

    def extract_with_default(self, response, css_selector, default="NA", strip=False):
        value = response.css(css_selector).get()
        if value:
            return value.strip() if strip else value
        else:
            logger.warning(f'Missing value for selector: {css_selector} on page: {response.url}')
            return default
        
    def save_file(self, response):
        country_name = response.meta['country_name']
        item = response.meta['item']
        file_name = response.url.split("/")[-1]
        file_content = response.body
        
        if file_name.endswith('.xlsx'):
            item['xlsx_file'] = {'file_name': file_name, 'file_content': file_content}
        elif file_name.endswith('.png'):
            item['png_file'] = {'file_name': file_name, 'file_content': file_content}
        
        logger.info(f'Saved file {file_name} for country {country_name}')
        yield item

    def save_image(self, response):
        country_name = response.meta['country_name']
        item = response.meta['item']
        file_name = response.url.split("/")[-1]
        file_content = response.body

        if file_name.endswith('.png'):
            item['images'].append({'file_name': file_name, 'file_content': file_content})

        logger.info(f'Saved image {file_name} for country {country_name}')
        yield item





