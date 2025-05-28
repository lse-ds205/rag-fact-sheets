import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


HOPRAG_PATTERNS = {
            'SUPPORTS': [
                (r'\d+(\.\d+)?\s*(MtCO2e|%|GW|MW|billion|million)', r'target|goal|commitment|reduce|achieve'),
                (r'invest.*\$?\d+.*billion', r'achieve|implement|deploy|fund'),
                (r'solar|wind|renewable.*\d+', r'emission.*\reduction|target|goal'),
                (r'carbon tax.*\d+', r'revenue|fund|support|finance'),
                (r'efficiency.*\d+.*%', r'reduction|saving|target')
            ],
            'EXPLAINS': [
                (r'NDC.*Nationally Determined Contribution', r'NDC(?!\w)'),
                (r'GHG.*greenhouse gas', r'GHG(?!\w)'),
                (r'carbon tax.*mechanism.*price', r'carbon tax'),
                (r'renewable energy.*includes.*solar.*wind', r'renewable'),
                (r'adaptation.*refers to|means', r'adaptation'),
                (r'mitigation.*refers to|means', r'mitigation')
            ],
            'CONTRADICTS': [
                (r'target.*\d+.*MtCO2e', r'target.*\d+.*MtCO2e'),
                (r'by \d{4}', r'by \d{4}'),
                (r'increase.*emissions', r'reduce.*emissions'),
                (r'not.*feasible|impossible', r'will.*implement|committed')
            ],
            'FOLLOWS': [
                (r'phase 1|first phase', r'phase 2|second phase|next phase'),
                (r'by 2030', r'after 2030|post-2030|2035|2040|2050'),
                (r'short.?term', r'medium.?term|long.?term'),
                (r'pilot|trial', r'scale.*up|full.*deployment')
            ],
            'CAUSES': [
                (r'due to|because of|result of', r'therefore|thus|consequently'),
                (r'leads to|results in|causes', r'impact|effect|consequence'),
                (r'if.*then', r'will.*result|outcome')
            ]
        }

class QueryType(Enum):
    """Enumeration of different query types for NDC documents"""
    NDC_TARGETS = "ndc_targets"
    EMISSIONS_REDUCTION = "emissions_reduction"
    RENEWABLE_ENERGY = "renewable_energy"
    ADAPTATION = "adaptation"
    MITIGATION = "mitigation"
    FINANCE = "finance"
    TECHNOLOGY = "technology"
    CAPACITY_BUILDING = "capacity_building"
    TRANSPORT = "transport"
    AGRICULTURE = "agriculture"
    FORESTRY = "forestry"
    WASTE = "waste"
    INDUSTRY = "industry"
    BUILDINGS = "buildings"
    ENERGY_EFFICIENCY = "energy_efficiency"
    CARBON_PRICING = "carbon_pricing"
    POLICY_MEASURES = "policy_measures"
    MONITORING = "monitoring"
    REPORTING = "reporting"
    INTERNATIONAL_COOPERATION = "international_cooperation"

@dataclass
class KeywordGroup:
    """Group of related keywords with boost weights"""
    primary_keywords: List[str]  # High importance keywords
    secondary_keywords: List[str]  # Medium importance keywords
    contextual_keywords: List[str]  # Context-providing keywords
    primary_weight: float = 2.0
    secondary_weight: float = 1.5
    contextual_weight: float = 1.2

class NDCKeywordMapper:
    """Sophisticated keyword mapping system for NDC document queries"""
    
    def __init__(self):
        self.keyword_mappings = self._initialize_keyword_mappings()
        self.compiled_patterns = self._compile_patterns()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_keyword_mappings(self) -> Dict[QueryType, KeywordGroup]:
        """Initialize comprehensive keyword mappings for different query types"""
        return {
            QueryType.NDC_TARGETS: KeywordGroup(
                primary_keywords=[
                    "reduction", "target", "goal", "objective", "commitment", "pledge",
                    "baseline", "compared to", "below", "decrease", "cut", "cutoff",
                    "unconditional", "conditional", "mitigation target"
                ],
                secondary_keywords=[
                    "percent", "%", "percentage", "by 2030", "by 2025", "by 2035",
                    "GHG", "greenhouse gas", "CO2", "carbon dioxide", "emissions intensity",
                    "BAU", "business as usual", "reference scenario"
                ],
                contextual_keywords=[
                    "NDC", "nationally determined contribution", "paris agreement",
                    "implementation", "achieve", "ambitious", "economy-wide"
                ]
            ),
            
            QueryType.EMISSIONS_REDUCTION: KeywordGroup(
                primary_keywords=[
                    "emissions reduction", "GHG reduction", "carbon reduction", "decarbonization",
                    "net zero", "carbon neutral", "emission intensity", "absolute reduction",
                    "per capita emissions", "sectoral reduction"
                ],
                secondary_keywords=[
                    "scope 1", "scope 2", "scope 3", "direct emissions", "indirect emissions",
                    "methane", "CH4", "nitrous oxide", "N2O", "fluorinated gases",
                    "carbon sink", "carbon sequestration"
                ],
                contextual_keywords=[
                    "measurement", "quantification", "inventory", "baseline year",
                    "reference level", "trajectory", "pathway"
                ]
            ),
            
            QueryType.RENEWABLE_ENERGY: KeywordGroup(
                primary_keywords=[
                    "renewable energy", "solar", "wind", "hydroelectric", "geothermal",
                    "biomass", "bioenergy", "renewable electricity", "clean energy",
                    "green energy", "sustainable energy"
                ],
                secondary_keywords=[
                    "photovoltaic", "PV", "wind turbine", "wind farm", "solar farm",
                    "hydropower", "mini-grid", "off-grid", "grid-connected",
                    "energy storage", "battery", "pumped hydro"
                ],
                contextual_keywords=[
                    "capacity", "MW", "GW", "generation", "penetration", "share",
                    "deployment", "installation", "development", "investment"
                ]
            ),
            
            QueryType.ADAPTATION: KeywordGroup(
                primary_keywords=[
                    "adaptation", "climate resilience", "resilient", "vulnerability",
                    "climate risk", "climate impact", "climate change adaptation",
                    "adaptive capacity", "disaster risk reduction"
                ],
                secondary_keywords=[
                    "sea level rise", "extreme weather", "drought", "flood", "storm",
                    "temperature increase", "precipitation change", "ecosystem adaptation",
                    "infrastructure adaptation", "coastal protection"
                ],
                contextual_keywords=[
                    "assessment", "planning", "early warning", "ecosystem services",
                    "traditional knowledge", "community-based", "nature-based solutions"
                ]
            ),
            
            QueryType.FINANCE: KeywordGroup(
                primary_keywords=[
                    "climate finance", "funding", "investment", "financial support",
                    "climate investment", "green finance", "carbon finance",
                    "climate fund", "international finance"
                ],
                secondary_keywords=[
                    "USD", "million", "billion", "cost", "budget", "expenditure",
                    "grant", "loan", "concessional", "blended finance",
                    "private sector", "public sector", "multilateral"
                ],
                contextual_keywords=[
                    "mobilization", "leveraging", "scaling up", "access",
                    "financial instrument", "mechanism", "facility"
                ]
            ),
            
            QueryType.TECHNOLOGY: KeywordGroup(
                primary_keywords=[
                    "technology transfer", "clean technology", "green technology",
                    "climate technology", "innovation", "research and development",
                    "technological capacity", "technology deployment"
                ],
                secondary_keywords=[
                    "carbon capture", "CCUS", "energy efficiency technology",
                    "smart grid", "electric vehicle", "hydrogen", "fuel cell",
                    "energy management system", "monitoring technology"
                ],
                contextual_keywords=[
                    "knowledge sharing", "technical assistance", "capacity enhancement",
                    "demonstration project", "pilot project", "scaling up"
                ]
            ),
            
            QueryType.TRANSPORT: KeywordGroup(
                primary_keywords=[
                    "transport", "transportation", "mobility", "vehicle", "automotive",
                    "public transport", "mass transit", "sustainable transport",
                    "low-carbon transport"
                ],
                secondary_keywords=[
                    "electric vehicle", "EV", "hybrid", "biofuel", "alternative fuel",
                    "fuel efficiency", "modal shift", "freight", "logistics",
                    "bus rapid transit", "BRT", "railway", "metro"
                ],
                contextual_keywords=[
                    "urban planning", "infrastructure", "fleet", "emission standard",
                    "fuel economy", "congestion", "accessibility"
                ]
            ),
            
            QueryType.ENERGY_EFFICIENCY: KeywordGroup(
                primary_keywords=[
                    "energy efficiency", "energy saving", "energy conservation",
                    "efficiency improvement", "energy performance", "efficient use",
                    "demand-side management", "energy intensity reduction"
                ],
                secondary_keywords=[
                    "building efficiency", "industrial efficiency", "appliance efficiency",
                    "lighting", "HVAC", "insulation", "smart building",
                    "energy audit", "efficiency standard", "labeling"
                ],
                contextual_keywords=[
                    "retrofit", "renovation", "upgrade", "optimization",
                    "best practices", "code", "regulation", "incentive"                ]
            )
        }
        
    def _compile_patterns(self) -> Dict[QueryType, Dict[str, re.Pattern]]:
        """Compile regex patterns for efficient matching"""
        compiled = {}
        
        for query_type, keyword_group in self.keyword_mappings.items():
            compiled[query_type] = {
                'primary': self._compile_pattern_from_keywords(keyword_group.primary_keywords),
                'secondary': self._compile_pattern_from_keywords(keyword_group.secondary_keywords),
                'contextual': self._compile_pattern_from_keywords(keyword_group.contextual_keywords)
            }
        
        return compiled
    
    def _compile_pattern_from_keywords(self, keywords: List[str]) -> re.Pattern:
        """Compile a list of keywords into a single regex pattern with proper word boundaries"""
        if not keywords:
            return re.compile(r'(?!.*)', re.IGNORECASE)  # Empty pattern that won't match anything
        
        # Escape special regex characters and create word boundaries
        escaped_keywords = []
        for keyword in keywords:
            # Handle multi-word phrases and special characters
            if ' ' in keyword or '-' in keyword:
                # For phrases, use word boundaries at the start and end
                escaped = re.escape(keyword)
                escaped_keywords.append(f"\\b{escaped}\\b")
            else:
                # For single words, use standard word boundaries
                escaped_keywords.append(f"\\b{re.escape(keyword)}\\b")
        
        pattern = '|'.join(escaped_keywords)
        return re.compile(pattern, re.IGNORECASE)
    
    def detect_query_type(self, query: str) -> List[Tuple[QueryType, float]]:
        """Detect the most likely query types based on query content"""
        query_lower = query.lower()
        scores = {}
        
        # Keywords that strongly indicate specific query types
        type_indicators = {
            'target': [QueryType.NDC_TARGETS],
            'goal': [QueryType.NDC_TARGETS],
            'reduction': [QueryType.EMISSIONS_REDUCTION, QueryType.NDC_TARGETS],
            'renewable': [QueryType.RENEWABLE_ENERGY],
            'solar': [QueryType.RENEWABLE_ENERGY],
            'wind': [QueryType.RENEWABLE_ENERGY],
            'adaptation': [QueryType.ADAPTATION],
            'resilience': [QueryType.ADAPTATION],
            'finance': [QueryType.FINANCE],
            'funding': [QueryType.FINANCE],
            'technology': [QueryType.TECHNOLOGY],
            'transport': [QueryType.TRANSPORT],
            'vehicle': [QueryType.TRANSPORT],
            'efficiency': [QueryType.ENERGY_EFFICIENCY],
            'building': [QueryType.BUILDINGS, QueryType.ENERGY_EFFICIENCY],
            'forest': [QueryType.FORESTRY],
            'agriculture': [QueryType.AGRICULTURE],
            'waste': [QueryType.WASTE],
            'industry': [QueryType.INDUSTRY],
            'carbon pricing': [QueryType.CARBON_PRICING],
            'policy': [QueryType.POLICY_MEASURES],
            'monitoring': [QueryType.MONITORING],
            'reporting': [QueryType.REPORTING]
        }
        
        # Score based on indicator keywords
        for indicator, query_types in type_indicators.items():
            if indicator in query_lower:
                for qt in query_types:
                    scores[qt] = scores.get(qt, 0) + 2.0
        
        # If no clear indicators, return top query types with lower confidence
        if not scores:
            return [(QueryType.NDC_TARGETS, 1.0), (QueryType.EMISSIONS_REDUCTION, 0.8)]
        
        # Sort by score and return top candidates
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:3]  # Return top 3 most likely 
    
    def get_relevant_keywords(self, query: str) -> Dict[str, Dict[str, List[str]]]:
        """Get relevant keywords for a given query"""
        detected_types = self.detect_query_type(query)
        relevant_keywords = {}
        
        for query_type, confidence in detected_types[:2]:  # Top 2 types
            if query_type in self.keyword_mappings:
                keyword_group = self.keyword_mappings[query_type]
                relevant_keywords[query_type.value] = {
                    'primary': keyword_group.primary_keywords,
                    'secondary': keyword_group.secondary_keywords,
                    'contextual': keyword_group.contextual_keywords
                }
    
        return relevant_keywords
    
    def evaluate_chunk_keywords(self, chunk_content: str, query: str) -> Dict[str, Any]:
        """
        Evaluate how well a chunk matches query-relevant keywords
        
        Args:
            chunk_content: Text content of the chunk
            query: User query
            
        Returns:
            Dictionary containing match scores and details
        """
        detected_types = self.detect_query_type(query)
        
        total_score = 0.0
        total_matches = 0
        match_breakdown = {}
        
        for query_type, type_confidence in detected_types:
            if query_type in self.compiled_patterns:
                patterns = self.compiled_patterns[query_type]
                keyword_group = self.keyword_mappings[query_type]
                
                # Count matches for each keyword category
                primary_matches = len(patterns['primary'].findall(chunk_content))
                secondary_matches = len(patterns['secondary'].findall(chunk_content))
                contextual_matches = len(patterns['contextual'].findall(chunk_content))
                
                # Calculate weighted score for this query type
                type_score = (
                    primary_matches * keyword_group.primary_weight +
                    secondary_matches * keyword_group.secondary_weight +
                    contextual_matches * keyword_group.contextual_weight
                ) * type_confidence
                
                total_score += type_score
                total_matches += primary_matches + secondary_matches + contextual_matches
                
                # Store detailed breakdown
                if primary_matches + secondary_matches + contextual_matches > 0:
                    match_breakdown[query_type.value] = {
                        'primary_matches': primary_matches,
                        'secondary_matches': secondary_matches,
                        'contextual_matches': contextual_matches,
                        'type_score': type_score,
                        'type_confidence': type_confidence
                    }
        
        return {
            'total_score': total_score,
            'total_matches': total_matches,
            'match_breakdown': match_breakdown,
            'detected_types': [{'type': qt.value, 'confidence': conf} for qt, conf in detected_types]
        }


# Legacy regex patterns for backward compatibility
"""
NDC Targets and Commitments: Matches emission reduction targets and commitments
"""
REGEX_NDC_TARGETS = r"\b(?:reduction|target|goal|objective|commitment|pledge|baseline|compared\s+to|below|decrease|cut|unconditional|conditional)\b"

"""
Emissions and GHG patterns: Matches greenhouse gas and emissions-related terms
"""
REGEX_EMISSIONS = r"\b(?:emissions?|GHG|greenhouse\s+gas|CO2|carbon\s+dioxide|methane|CH4|nitrous\s+oxide|N2O)\b"

"""
Renewable Energy patterns: Matches renewable energy technologies and terms
"""
REGEX_RENEWABLE_ENERGY = r"\b(?:renewable\s+energy|solar|wind|hydroelectric|geothermal|biomass|bioenergy|clean\s+energy)\b"

"""
Climate Finance patterns: Matches financial terms related to climate action
"""
REGEX_CLIMATE_FINANCE = r"\b(?:climate\s+finance|funding|investment|financial\s+support|USD|million|billion|grant|loan)\b"

"""
Numerical values with units: Extracts numerical targets and measurements
"""
REGEX_NUMERICAL_TARGETS = r"\b\d+(?:\.\d+)?(?:\s*%|\s*percent|\s*MW|\s*GW|\s*Mt|\s*Gt)\b"

"""
Policy and implementation terms: Matches policy-related language
"""
REGEX_POLICY_MEASURES = r"\b(?:policy|measure|action|plan|strategy|framework|regulation|standard|incentive)\b"

# Cleaning patterns
"""
Remove excessive whitespace and normalize text
"""
REGEX_NORMALIZE_WHITESPACE = r"\s+"

"""
Remove special characters that might interfere with search
"""
REGEX_REMOVE_SPECIAL_CHARS = r"[^\w\s\-\.,;:!?\(\)\[\]{}\"']"

"""
Extract clean sentences for better processing
"""
REGEX_SENTENCE_BOUNDARIES = r"(?<=[.!?])\s+(?=[A-Z])"

# Global instance of NDCKeywordMapper for use across the application
ndc_keyword_mapper = NDCKeywordMapper()