import scrapy

class ClimateActionTrackerSpider(scrapy.Spider):
    name = "climate_action_tracker"
    allowed_domains = ["climateactiontracker.org"]
    start_urls = ["https://climateactiontracker.org/countries/turkey/"]

    def parse(self, response):
        """Extract country name, flag, and rating, then proceed to policies & actions."""
        country_name = response.css('h1::text').get()
        overall_rating = response.css('.ratings-matrix__overall dd::text').get()
        flag_url = response.css('.headline__flag img::attr(src)').get()

        if flag_url:
            flag_url = f"https://climateactiontracker.org{flag_url}"

        # Extract image URLs
        low_res_image = response.css('div[data-component-graph-embed]::attr(data-props-graph-image-url)').get()
        high_res_image = response.css('div[data-component-graph-embed]::attr(data-props-graph-hires-image-url)').get()
        base_url = "https://climateactiontracker.org"
        if low_res_image and not low_res_image.startswith("http"):
            low_res_image = base_url + low_res_image
        if high_res_image and not high_res_image.startswith("http"):
            high_res_image = base_url + high_res_image

        yield {
            'country_name': country_name,
            'overall_rating': overall_rating,
            'flag_url': flag_url,
            'low_res_image_url': low_res_image,
            'high_res_image_url': high_res_image
        }

        # Extract policies & action page URL
        policies_action_url = response.url + "policies-action/"
        yield scrapy.Request(url=policies_action_url, callback=self.parse_policies_action, meta={'country_name': country_name})

    def parse_policies_action(self, response):
        """Extract policies & action data, ensuring subheadings & bullet points are placed correctly."""
        country_name = response.meta['country_name']
        policies_data = []

        sections = response.css('div.content-section')
        current_heading = None
        content_accumulator = []

        for section in sections:
            main_heading = section.css('h3::text').get()

            for paragraph in section.css('div.content-section__content p, ul'):
                # Check for subheadings (bold elements alone in a paragraph)
                bold_subheadings = paragraph.css('strong::text, b::text').getall()
                paragraph_texts = paragraph.css('::text').getall()

                # Hard split when a bold subheading appears
                if bold_subheadings and len(paragraph_texts) == len(bold_subheadings):
                    if current_heading and content_accumulator:
                        policies_data.append({'heading': current_heading, 'content': " ".join(content_accumulator)})
                        content_accumulator = []

                    # Set new heading as bold subheading
                    current_heading = " ".join([text.strip() for text in bold_subheadings if text.strip()])
                
                else:
                    text_content = " ".join([text.strip() for text in paragraph_texts if text.strip()])
                    content_accumulator.append(text_content)

                # Handle bullet points
                for bullet in paragraph.css('li'):
                    bullet_text = bullet.css('::text, strong::text, b::text').getall()
                    formatted_bullet = "• " + " ".join([text.strip() for text in bullet_text if text.strip()])
                    content_accumulator.append(formatted_bullet)

            if main_heading and not current_heading:
                current_heading = main_heading.strip()

        # Extract Land Use & Forestry section
        land_use_section = response.css('div#section__not-significant')
        land_use_heading = land_use_section.css('dt::text').get()
        land_use_content = land_use_section.css('dd::text').get()

        if land_use_heading and land_use_content:
            policies_data.append({
                'heading': land_use_heading.strip(),
                'content': land_use_content.strip()
            })

        # Append the last collected section
        if current_heading and content_accumulator:
            policies_data.append({'heading': current_heading, 'content': " ".join(content_accumulator)})

        yield {
            'country_name': country_name,
            'policies_action_sections': policies_data
        }
