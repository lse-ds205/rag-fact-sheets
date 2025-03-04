import scrapy

class ClimateTrackerItem(scrapy.Item):
    country_name = scrapy.Field()
    overall_rating = scrapy.Field()
    
    # Updated fields based on website structure
    policies_action_modelled_pathways = scrapy.Field()
    ndc_target_modelled_pathways = scrapy.Field()
    ndc_target_fair_share = scrapy.Field()
    climate_finance = scrapy.Field()
    net_zero_target = scrapy.Field()
