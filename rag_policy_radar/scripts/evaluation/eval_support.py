import pandas as pd
from typing import List, Dict, Any

from scripts.climate_policy_pipelines.cp1.pipeline import (
    run_cp1a_assessment, 
    run_cp1a_assessment_large_context, 
    run_cp1b_assessment
)

# Create mapping from full country names to country codes
country_name_to_code = {
    'United States': 'USA',
    'United Kingdom': 'GBR',
    'Germany': 'DEU',
    'France': 'FRA',
    'Japan': 'JPN',
    'Canada': 'CAN',
    'Australia': 'AUS',
    'Brazil': 'BRA',
    'China': 'CHN',
    'India': 'IND',
    'South Africa': 'ZAF',
    'Mexico': 'MEX',
    'Argentina': 'ARG',
    'Chile': 'CHL',
    'Indonesia': 'IDN',
    'South Korea': 'KOR',
    'Turkey': 'TUR',
    'Russia': 'RUS',
    'Italy': 'ITA',
    'Spain': 'ESP',
    'Netherlands': 'NLD',
    'Sweden': 'SWE',
    'Norway': 'NOR',
    'Denmark': 'DNK',
    'Finland': 'FIN',
    'Switzerland': 'CHE',
    'Austria': 'AUT',
    'Belgium': 'BEL',
    'Poland': 'POL',
    'Czech Republic': 'CZE',
    'Hungary': 'HUN',
    'Portugal': 'PRT',
    'Greece': 'GRC',
    'Ireland': 'IRL',
    'New Zealand': 'NZL',
    'Israel': 'ISR',
    'Thailand': 'THA',
    'Malaysia': 'MYS',
    'Singapore': 'SGP',
    'Philippines': 'PHL',
    'Vietnam': 'VNM',
    'Egypt': 'EGY',
    'Nigeria': 'NGA',
    'Kenya': 'KEN',
    'Morocco': 'MAR',
    'Colombia': 'COL',
    'Peru': 'PER',
    'Ecuador': 'ECU',
    'Venezuela': 'VEN',
    'Uruguay': 'URY',
    'Paraguay': 'PRY'
}

def evaluate_cp1_assessments(
    countries: List[str], 
    ascor_ground_truth: pd.DataFrame,
    ascor_column_name: str = 'Assessment_Column'
) -> pd.DataFrame:
    """
    Evaluate CP1 assessments against ASCOR ground truth for multiple countries.
    
    Args:
        countries: List of country names to evaluate
        ascor_ground_truth: DataFrame containing ASCOR ground truth data
        ascor_column_name: Name of the column containing ASCOR assessment results
        
    Returns:
        DataFrame with comparison results
    """
    results_data = []
    
    for country in countries:
        # Get ASCOR ground truth for this country
        ascor_result = ascor_ground_truth[ascor_ground_truth["Country"] == country]

        # Convert country name to country code for the assessment functions
        country_code = country_name_to_code.get(country)
        
        if country_code is None:
            print(f"Warning: No country code mapping found for '{country}'. Skipping...")
            continue
        
        # Run CP1A assessment using country code
        cp1a_rag = run_cp1a_assessment(country=country_code, detailed=False)
        cp1a_large_context_rag = run_cp1a_assessment_large_context(
            country_code, detailed=False, print_results=False
        )
        cp1b_rag = run_cp1b_assessment(
            country_code, detailed=False, print_results=False
        )
        
        # Store results
        results_data.append({
            'Country': country,
            'ASCOR_CP1A': ascor_result.iloc[0]["indicator CP.1.a"],
            'ASCOR_CP1B': ascor_result.iloc[0]["indicator CP.1.b"],
            'CP1A_Assessment': cp1a_rag,
            'CP1A_Large_Context_Assessment': cp1a_large_context_rag,
            'CP1B_Assessment': cp1b_rag
        })
                        
    comparison_df = pd.DataFrame(results_data)    
    return comparison_df