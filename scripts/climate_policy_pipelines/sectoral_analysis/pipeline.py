
import os
from openai import OpenAI

import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import BaseOutputParser
import json
from dotenv import load_dotenv

load_dotenv()


from typing_extensions import Annotated, TypedDict
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from typing import Dict, Any
import json

from scripts.climate_policy_pipelines.shared.llm_models import (
    super_basic_model,
    standard_model,
    large_context_llm)

from scripts.climate_policy_pipelines.sectoral_analysis.classes import (
    NetZeroTargetDetails,
    NetZeroAssessment,
    PeriodAnalysis1970s,
    PeriodAnalysis1990s,
    PeriodAnalysis2010s,
    LegislativeTimeline,
    LegislativeSummary,
    SimpleASCORReport
)

from scripts.retrieval.retrieval_pipeline import do_retrieval

def _format_context(chunks):
    """Format retrieved chunks into readable context string"""
    context_parts = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            content = chunk.get('chunk_content', str(chunk))
            doc_info = chunk.get('document', 'Unknown document')
            page_info = chunk.get('page_number', 'Unknown page')
            context_parts.append(f"[Source {i+1}: {doc_info}, Page {page_info}]\n{content}")
        else:
            context_parts.append(f"[Source {i+1}]\n{str(chunk)}")
    
    return "\n\n".join(context_parts)

def generate_net_zero_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate context specifically for net zero analysis"""
    search_query = f"net zero targets {inputs['country']} {inputs['sector']} commitments legal binding"
    retrieved_chunks = do_retrieval(search_query, k=10)
    context = _format_context(retrieved_chunks)
    
    return {
        **inputs,
        "net_zero_context": context
    }

def generate_period_contexts(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Generate contexts for all legislative periods"""
    country = inputs['country']
    sector = inputs['sector']
    
    # Generate context for each period
    contexts = {}
    
    # 1970s-1980s context
    search_1970s = f"{country} {sector} legislation 1970s 1980s policy frameworks regulations"
    chunks_1970s = do_retrieval(search_1970s, k=8)
    contexts["context_1970s_1980s"] = _format_context(chunks_1970s)
    
    # 1990s-2000s context
    search_1990s = f"{country} {sector} legislation 1990s 2000s policy frameworks regulations"
    chunks_1990s = do_retrieval(search_1990s, k=8)
    contexts["context_1990s_2000s"] = _format_context(chunks_1990s)
    
    # 2010s-present context
    search_2010s = f"{country} {sector} legislation 2010s 2020s policy frameworks regulations recent"
    chunks_2010s = do_retrieval(search_2010s, k=8)
    contexts["context_2010s_present"] = _format_context(chunks_2010s)
    
    return {
        **inputs,
        **contexts
    }

def analyze_net_zero(inputs: Dict[str, Any]) -> NetZeroAssessment:
    """Dedicated LLM for net zero analysis - uses larger context model for complexity"""
    llm = large_context_llm.with_structured_output(NetZeroAssessment)
    
    prompt = f"""Analyze net zero targets for {inputs['country']} in {inputs['sector']} sector.
    
    Context: {inputs.get('net_zero_context', 'No specific context provided')}
    
    Provide:
    1. One paragraph analysis of net zero commitments
    2. Structured target details (date, scope, legal status)
    3. Confidence assessment
    
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).

    """
    
    return llm.invoke(prompt)

def analyze_period_1970s_1980s(inputs: Dict[str, Any]) -> PeriodAnalysis1970s:
    """Dedicated LLM for 1970s-1980s legislative analysis"""
    prompt = f"""Analyze 1970s-1980s legislation for {inputs['country']} {inputs['sector']} sector.
    Focus on major policy shifts, regulatory frameworks shifts, and legislative milestones relevant to climate policy in the sector.
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).
    Context: {inputs.get('context_1970s_1980s', '')}
    """
    
    result = super_basic_model.invoke(prompt)
    return PeriodAnalysis1970s(period_1970s_1980s=result.content)

def analyze_period_1990s_2000s(inputs: Dict[str, Any]) -> PeriodAnalysis1990s:
    """Dedicated LLM for 1990s-2000s legislative analysis"""
    prompt = f"""Analyze 1990s-2000s legislation for {inputs['country']} {inputs['sector']} sector.
    Focus on major policy shifts, regulatory frameworks shifts, and legislative milestones relevant to climate policy in the sector.
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).
    Context: {inputs.get('context_1990s_2000s', '')}
    """
    
    result = standard_model.invoke(prompt)
    return PeriodAnalysis1990s(period_1990s_2000s=result.content)

def analyze_period_2010s_present(inputs: Dict[str, Any]) -> PeriodAnalysis2010s:
    """Dedicated LLM for 2010s-Present legislative analysis - uses larger model for recent complex policies"""
    prompt = f"""Analyze 2010s-Present legislation for {inputs['country']} {inputs['sector']} sector.
    Focus on major policy shifts, regulatory frameworks shifts, and legislative milestones relevant to climate policy in the sector.
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).
    Context: {inputs.get('context_2010s_present', '')}
    """
    
    result = large_context_llm.invoke(prompt)
    return PeriodAnalysis2010s(period_2010s_present=result.content)

def synthesize_legislative_timeline(timeline_parts: Dict[str, Any]) -> LegislativeTimeline:
    """Combine timeline periods into structured output"""
    return LegislativeTimeline(
        period_1970s_1980s=timeline_parts["period_1970s_1980s"],
        period_1990s_2000s=timeline_parts["period_1990s_2000s"],
        period_2010s_present=timeline_parts["period_2010s_present"]
    )

def synthesize_legislative_summary(inputs: Dict[str, Any]) -> LegislativeSummary:
    """Dedicated LLM for synthesizing timeline into coherent summary - uses standard model for synthesis"""
    llm = standard_model.with_structured_output(LegislativeSummary)
    
    timeline = inputs["legislative_timeline"]
    prompt = f"""Synthesize 50 years of legislative evolution for {inputs['country']} {inputs['sector']}.
    
    Timeline data:
    - 1970s-1980s: {timeline['period_1970s_1980s']}
    - 1990s-2000s: {timeline['period_1990s_2000s']}
    - 2010s-Present: {timeline['period_2010s_present']}
    
    Create one flowing paragraph showing policy evolution and assess confidence.
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).
    """
    
    return llm.invoke(prompt)

def assemble_final_report(inputs: Dict[str, Any]) -> SimpleASCORReport:
    """Assemble all components into final report structure"""
    return SimpleASCORReport(
        country=inputs["country"],
        sector=inputs["sector"],
        net_zero_response=inputs["net_zero_assessment"]["net_zero_response"],
        net_zero_details=inputs["net_zero_assessment"]["net_zero_details"],
        net_zero_confidence=inputs["net_zero_assessment"]["net_zero_confidence"],
        legislation_response=inputs["legislative_summary"]["legislation_response"],
        legislative_timeline=inputs["legislative_timeline"],
        legislation_confidence=inputs["legislative_summary"]["legislation_confidence"]
    )

# Main Workflow Chain
def create_ascor_workflow():
    """Create the complete ASCOR report generation workflow"""
    
    # Context generation chain
    context_chain = (
        RunnablePassthrough()
        | RunnableLambda(generate_net_zero_context)
        | RunnableLambda(generate_period_contexts)
    )
    
    # Parallel legislative analysis for each time period
    legislative_chain = (
        RunnablePassthrough()
        | {
            "period_1970s_1980s": RunnableLambda(analyze_period_1970s_1980s),
            "period_1990s_2000s": RunnableLambda(analyze_period_1990s_2000s), 
            "period_2010s_present": RunnableLambda(analyze_period_2010s_present)
        }
        | RunnableLambda(lambda x: {
            "period_1970s_1980s": x["period_1970s_1980s"].period_1970s_1980s,
            "period_1990s_2000s": x["period_1990s_2000s"].period_1990s_2000s,
            "period_2010s_present": x["period_2010s_present"].period_2010s_present
        })
        | RunnableLambda(synthesize_legislative_timeline)
    )
    
    # Complete workflow
    workflow = (
        RunnablePassthrough()
        | context_chain  # Generate all contexts first
        | {
            "country": lambda x: x["country"],
            "sector": lambda x: x["sector"], 
            "net_zero_assessment": RunnableLambda(analyze_net_zero),
            "legislative_timeline": legislative_chain,
            # Pass through context data for debugging/reference
            "net_zero_context": lambda x: x.get("net_zero_context", ""),
            "context_1970s_1980s": lambda x: x.get("context_1970s_1980s", ""),
            "context_1990s_2000s": lambda x: x.get("context_1990s_2000s", ""),
            "context_2010s_present": lambda x: x.get("context_2010s_present", "")
        }
        | {
            "country": lambda x: x["country"],
            "sector": lambda x: x["sector"],
            "net_zero_assessment": lambda x: x["net_zero_assessment"],
            "legislative_timeline": lambda x: x["legislative_timeline"],
            "legislative_summary": RunnableLambda(synthesize_legislative_summary)
        }
        | RunnableLambda(assemble_final_report)
    )
    
    return workflow

# Usage
def generate_ascor_report(country: str, sector: str, context: str = "") -> SimpleASCORReport:
    """Generate ASCOR report for given country and sector"""
    workflow = create_ascor_workflow()
    
    inputs = {
        "country": country,
        "sector": sector,
        "context": context
    }
    
    return workflow.invoke(inputs)

def format_report_as_markdown(report: SimpleASCORReport) -> str:
    """Convert the structured report to formatted markdown"""
    
    markdown = f"""# ASCOR Sectoral Transition Report

## Executive Summary
**Country:** {report['country']}  
**Sector:** {report['sector']}  
---

## Key Findings

### 1. Net Zero Target Assessment

**Question:** Have they set a net zero target?

{report['net_zero_response']}

**Key Details:**
- Target Date: {report['net_zero_details']['target_date']}
- Scope: {report['net_zero_details']['scope']}
- Legal Status: {report['net_zero_details']['legal_status']}

**Confidence in Assessment:** {report['net_zero_confidence']}

### 2. Legislative Evolution Assessment

**Question:** How has the legislation of this country changed in the past 50 years?

{report['legislation_response']}

**Legislative Timeline:**
- **1970s-1980s:** {report['legislative_timeline']['period_1970s_1980s']}
- **1990s-2000s:** {report['legislative_timeline']['period_1990s_2000s']}
- **2010s-Present:** {report['legislative_timeline']['period_2010s_present']}

**Confidence in Assessment:** {report['legislation_confidence']}
"""
    
    return markdown
