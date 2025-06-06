from typing_extensions import Annotated, TypedDict
from typing import Literal


class NetZeroTargetDetails(TypedDict):
    """Structured details for net zero target assessment"""
    target_date: Annotated[str, ..., "Target achievement date or 'Not Set'"]
    scope: Annotated[Literal["Economy-wide", "Sectoral", "Conditional"], ..., "Target scope classification"]
    legal_status: Annotated[Literal["Legally Binding", "Policy Commitment", "Aspirational"], ..., "Legal status of the target"]

class NetZeroAssessment(TypedDict):
    """Complete net zero target assessment from dedicated LLM"""
    net_zero_response: Annotated[str, ..., "One paragraph analyzing net zero commitments, including timeline, scope, and conditional aspects"]
    net_zero_details: Annotated[NetZeroTargetDetails, ..., "Structured key details about the net zero target"]
    net_zero_confidence: Annotated[Literal["High", "Medium", "Low"], ..., "Confidence level in the net zero assessment"]

class PeriodAnalysis1970s(TypedDict):
    """1970s-1980s legislative analysis result"""
    period_1970s_1980s: Annotated[str, ..., "Key legislative developments in 1970s-1980s period"]

class PeriodAnalysis1990s(TypedDict):
    """1990s-2000s legislative analysis result"""
    period_1990s_2000s: Annotated[str, ..., "Key legislative developments in 1990s-2000s period"]

class PeriodAnalysis2010s(TypedDict):
    """2010s-Present legislative analysis result"""
    period_2010s_present: Annotated[str, ..., "Key legislative developments in 2010s-Present period"]

class LegislativeTimeline(TypedDict):
    """Timeline breakdown of legislative changes from dedicated LLM"""
    period_1970s_1980s: Annotated[str, ..., "Key legislative developments in 1970s-1980s period"]
    period_1990s_2000s: Annotated[str, ..., "Key legislative developments in 1990s-2000s period"]  
    period_2010s_present: Annotated[str, ..., "Key legislative developments in 2010s-Present period"]

class LegislativeSummary(TypedDict):
    """Legislative evolution summary from synthesis LLM"""
    legislation_response: Annotated[str, ..., "One paragraph analyzing 50 years of legislative evolution, focusing on policy shifts and regulatory frameworks"]
    legislation_confidence: Annotated[Literal["High", "Medium", "Low"], ..., "Confidence level in the legislative assessment"]

class SimpleASCORReport(TypedDict):
    """Final assembled report structure"""
    
    # Executive Summary fields
    country: Annotated[str, ..., "Name of the country being analyzed"]
    sector: Annotated[str, ..., "Name of the sector being analyzed"]
    
    # Net Zero Target Assessment (from specialized LLM)
    net_zero_response: Annotated[str, ..., "One paragraph analyzing net zero commitments"]
    net_zero_details: Annotated[NetZeroTargetDetails, ..., "Structured key details about the net zero target"]
    net_zero_confidence: Annotated[Literal["High", "Medium", "Low"], ..., "Confidence level in the net zero assessment"]
    
    # Legislative Evolution Assessment (assembled from multiple LLMs)
    legislation_response: Annotated[str, ..., "One paragraph analyzing 50 years of legislative evolution"]
    legislative_timeline: Annotated[LegislativeTimeline, ..., "Structured timeline of legislative changes by period"]
    legislation_confidence: Annotated[Literal["High", "Medium", "Low"], ..., "Confidence level in the legislative assessment"]
