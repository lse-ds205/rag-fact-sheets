import pycountry

from scrapy.contracts import Contract
from scrapy.exceptions import ContractFail

class ValidCountryContract(Contract):
    """Check if country name is a valid country
    @valid_country
    """
    name = 'valid_country'

    def post_process(self, output):

        for item in output:
            search_country = pycountry.countries.get(name=item['country_name'])
            if not search_country:
                raise ContractFail(
                    f"Invalid country name: {item['country_name']}"
                )