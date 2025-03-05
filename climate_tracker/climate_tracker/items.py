import scrapy

class ClimateTrackerItem(scrapy.Item):
    country_name = scrapy.Field()
    overall_rating = scrapy.Field()
    flag_url = scrapy.Field()  # ✅ Stores the flag URL
    policies_action_content = scrapy.Field()  # ✅ Stores all paragraph text from policies & action page
    policies_action_headings = scrapy.Field()  # ✅ Stores all section headings from policies & action page
