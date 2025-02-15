from scrapy.contracts import Contract
from scrapy.exceptions import ContractFail

class RatingValidContract(Contract):
    """Check if rating is one of the valid options
    @valid_rating
    """
    name = 'valid_rating'

    def pre_process(self, output):
        print(output)


    def post_process(self, output):
        valid_ratings = [
            'Insufficient',
            'Compatible'
        ]
        
        print(output)

        for item in output:
            if 'overall_rating' not in item:
                raise ContractFail("Missing overall_rating field")
            
            if item['overall_rating'] not in valid_ratings:
                raise ContractFail(
                    f"Invalid rating: {item['overall_rating']}"
                ) 