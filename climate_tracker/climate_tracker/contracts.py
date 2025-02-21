import pycountry
from typing import Dict, List

from scrapy.contracts import Contract
from scrapy.exceptions import ContractFail
from scrapy.http import Request, Response

class ValidCountryContract(Contract):
    """Check if country name is a valid country
    @valid_country
    """
    name = 'valid_country'

    def post_process(self, output):
        for item in output:
            search_country = pycountry.countries.lookup(item['country_name'])
            if not search_country:
                raise ContractFail(f"Invalid country name: {item['country_name']}")

class ValidIndicatorsContract(Contract):
    """Check if indicators list has the correct structure
    @valid_indicators
    """
    name = 'valid_indicators'

    def post_process(self, output):

        for item in output:
            if 'indicators' not in item:
                raise ContractFail("Missing indicators list in output")
            
            if not isinstance(item['indicators'], list):
                raise ContractFail("Indicators must be a list")

            for indicator in item['indicators']:
                if 'term' not in indicator:
                    raise ContractFail(f"Missing 'term' in indicator: {indicator}")
                if 'value' not in indicator:
                    raise ContractFail(f"Missing 'value' in indicator: {indicator}")

class CompleteDataContract(Contract):
    """Check if all required fields are present and non-empty
    @complete_data
    """
    name = 'complete_data'

    def post_process(self, output):
        required_fields = ['country_name', 'overall_rating', 'flag_url', 'indicators']
        
        for item in output:
            for field in required_fields:
                if field not in item:
                    raise ContractFail(f"Missing required field: {field}")
                if not item[field]:
                    raise ContractFail(f"Empty required field: {field}")

            # Check first 3 indicators have complete data
            for i, indicator in enumerate(item['indicators']):
                if i < 3:  # First 3 indicators should have all fields
                    required_indicator_fields = ['term', 'term_details', 'value', 'metric']
                    for field in required_indicator_fields:
                        if field not in indicator:
                            raise ContractFail(f"Missing field {field} in indicator {i+1}")
                else:
                    # The fourth indicator might have missing fields
                    # TODO: Write a proper test for this too later
                    pass