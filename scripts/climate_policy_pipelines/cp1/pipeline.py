import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnableLambda
import json
from dotenv import load_dotenv
from langdetect import detect


load_dotenv()

from scripts.retrieval.retrieval_pipeline import do_retrieval

from scripts.climate_policy_pipelines.shared.llm_models import (
    llm,
    multilingual_llm,
    large_context_llm  
)

from scripts.climate_policy_pipelines.cp1.prompts import (
    cp1a_criterion_1_prompt,
    cp1a_criterion_2_prompt,
    cp1a_criterion_3_prompt,
    cp1a_criterion_4_prompt,
    cp1a_final_assessment_prompt,

    comprehensive_assessment_prompt,

    cp1b_criterion_1_prompt,
    cp1b_criterion_2_prompt,
    cp1b_criterion_3_prompt,
    cp1b_final_assessment_prompt
)

from scripts.climate_policy_pipelines.cp1.chains import (
    cp1a_criterion_1_chain,
    cp1a_criterion_2_chain,
    cp1a_criterion_3_chain,
    cp1a_criterion_4_chain,
    cp1a_criterion_1_chain_noneng,
    cp1a_criterion_2_chain_noneng,
    cp1a_criterion_3_chain_noneng,
    cp1a_criterion_4_chain_noneng,
    cp1b_criterion_1_chain,
    cp1b_criterion_2_chain,
    cp1b_criterion_3_chain,
    cp1b_criterion_1_chain_noneng,
    cp1b_criterion_2_chain_noneng,
    cp1b_criterion_3_chain_noneng
)

# CP1a assessment

def get_chunks_for_cp1a(country=None, k=20):
    """
    Get chunks specifically for CP1a assessment using predefined prompts
    
    Args:
        k (int): Number of chunks to retrieve for each criterion
    
    Returns:
        list: List of 4 elements [cp1a_1_content, cp1a_2_content, cp1a_3_content, cp1a_4_content]
    """
    # Define specific prompts for each CP1a criterion
    cp1a_prompts = [
        "strategic direction for decarbonisation Paris Agreement national long-term target",
        "climate law enshrined in law legally binding framework",
        "obligations carbon budgets emissions targets monitoring requirements",
        "environmental law executive climate strategy broad framework"
    ]
    
    # Get chunks for each criterion
    retrieved_chunks = []
    for prompt in cp1a_prompts:
        chunks = do_retrieval(prompt, country=country, k=k)
        retrieved_chunks.append(chunks)
    
    return retrieved_chunks


def detect_language(content):
    """Detect language of content (list of chunks or string)"""
    try:
        if isinstance(content, list):
            text = content[0]["chunk_content"] if content else ""
        else:
            text = str(content)
        return detect(text)
    except:
        return 'en'

def evaluate_all_criteria_multilingual(cp1a_retrieved_chunks, return_only_result=False):
    """Evaluate all four criteria using appropriate LLM based on language detection
    
    Args:
        retrieved_chunks: List of 4 elements [cp1a_1_content, cp1a_2_content, cp1a_3_content, cp1a_4_content]
        return_only_result: If True, returns only YES/NO, if False returns full details
    """
    
    # Extract individual content lists
    cp1a_1_content, cp1a_2_content, cp1a_3_content, cp1a_4_content = cp1a_retrieved_chunks
    
    # Detect languages for each context
    lang_1 = detect_language(cp1a_1_content)
    lang_2 = detect_language(cp1a_2_content)
    lang_3 = detect_language(cp1a_3_content)
    lang_4 = detect_language(cp1a_4_content)
    
    # Helper function to choose appropriate chain
    def get_chain_for_language(language, standard_chain, multilingual_chain):
        return multilingual_chain if language != 'en' else standard_chain
    
    # Route to appropriate chains based on language
    chain_1 = get_chain_for_language(lang_1, cp1a_criterion_1_chain, cp1a_criterion_1_chain_noneng)
    chain_2 = get_chain_for_language(lang_2, cp1a_criterion_2_chain, cp1a_criterion_2_chain_noneng)
    chain_3 = get_chain_for_language(lang_3, cp1a_criterion_3_chain, cp1a_criterion_3_chain_noneng)
    chain_4 = get_chain_for_language(lang_4, cp1a_criterion_4_chain, cp1a_criterion_4_chain_noneng)
    
    # Execute evaluations
    cp1a_criterion_1_result = chain_1.invoke({"context": cp1a_1_content})
    cp1a_criterion_2_result = chain_2.invoke({"context": cp1a_2_content})
    cp1a_criterion_3_result = chain_3.invoke({"context": cp1a_3_content})
    cp1a_criterion_4_result = chain_4.invoke({"context": cp1a_4_content})
    
    if return_only_result:
        # Return minimal structure for YES/NO chain
        return {
            "criterion_1_result": cp1a_criterion_1_result.content,
            "criterion_2_result": cp1a_criterion_2_result.content,
            "criterion_3_result": cp1a_criterion_3_result.content,
            "criterion_4_result": cp1a_criterion_4_result.content
        }
    else:
        # Return full details for analysis
        return {
            "criterion_1_result": cp1a_criterion_1_result.content,
            "criterion_2_result": cp1a_criterion_2_result.content,
            "criterion_3_result": cp1a_criterion_3_result.content,
            "criterion_4_result": cp1a_criterion_4_result.content,
            "languages_detected": {
                "context_1": lang_1,
                "context_2": lang_2,
                "context_3": lang_3,
                "context_4": lang_4
            }
        }

def extract_yes_no_result(assessment_result):
    """Extract just YES or NO from the final assessment result"""
    content = assessment_result.content.upper()
    if "YES" in content and "NO" not in content:
        return "YES"
    elif "NO" in content:
        return "NO"
    else:
        return "UNCLEAR"  # Fallback for edge cases

# Two separate chains for different use cases
# Full detailed assessment chain
cp1a_complete_assessment_chain_detailed = (
    RunnableLambda(lambda x: evaluate_all_criteria_multilingual(x, return_only_result=False))
    | cp1a_final_assessment_prompt 
    | llm
)

# Simple YES/NO assessment chain
cp1a_complete_assessment_chain_simple = (
    RunnableLambda(lambda x: evaluate_all_criteria_multilingual(x, return_only_result=True))
    | cp1a_final_assessment_prompt 
    | llm
    | RunnableLambda(extract_yes_no_result)
)


def run_cp1a_assessment(country=None, detailed=True):
    """
    Main function to run CP1a assessment with a single call
    
    Args:
        retrieved_chunks: List of 4 elements [cp1a_1_content, cp1a_2_content, cp1a_3_content, cp1a_4_content]
        detailed: If True, returns full details; if False, returns only YES/NO
    
    Returns:
        Assessment result (detailed dict or simple YES/NO string)
    """
    retrieved_chunks = get_chunks_for_cp1a(country)

    if detailed:
        result = cp1a_complete_assessment_chain_detailed.invoke(retrieved_chunks)
        return result
    else:
        result = cp1a_complete_assessment_chain_simple.invoke(retrieved_chunks)
        return result



# CPIa Large Context Assessment

def run_cp1a_assessment_large_context(country=None, print_results=True):
    """
    Run comprehensive CP1a framework climate law assessment using large context model
    
    Args:
        context_documents: List of document chunks or context data
        print_results: If True, prints formatted results to console
        
    Returns:
        LangChain result object with .content property containing formatted markdown assessment
    """
    context_documents = get_chunks_for_cp1a(country=country)

    # Create the single-model chain
    single_model_assessment_chain = comprehensive_assessment_prompt | large_context_llm
    
    # Run the assessment
    result = single_model_assessment_chain.invoke({"context": context_documents})
    
    if print_results:
        print("Large Context Assessment:")
        print(result.content)
    
    return result

# CP1b assessment

def get_chunks_for_cp1b(country=None, k=25):
    """
    Get chunks specifically for CP1b assessment using predefined prompts
    
    Args:
        country (str, optional): 3-letter country code
        k (int): Number of chunks to retrieve for each criterion
    
    Returns:
        list: List of 3 elements [cp1b_1_content, cp1b_2_content, cp1b_3_content]
    """
    # Define specific prompts for each CP1b criterion
    cp1b_prompts = [
        "accountability specification who accountable parliament executive authorities",
        "compliance assessment monitoring reporting verification parliamentary oversight",
        "non-compliance consequences parliamentary intervention judicial orders financial penalties"
    ]
    
    # Get chunks for each criterion - FIXED: now passes country parameter
    retrieved_chunks = []
    for prompt in cp1b_prompts:
        chunks = do_retrieval(prompt, country=country, k=k)  # Added country parameter
        retrieved_chunks.append(chunks)
    
    return retrieved_chunks

def evaluate_all_criteria_multilingual_cp1b(cp1b_retrieved_chunks, return_only_result=False):
    """Evaluate all three accountability criteria using appropriate LLM based on language detection
    
    Args:
        retrieved_chunks: List of 3 elements [cp1b_1_content, cp1b_2_content, cp1b_3_content]
        return_only_result: If True, returns only YES/NO, if False returns full details
    """
    
    # Extract individual content lists
    cp1b_1_content, cp1b_2_content, cp1b_3_content = cp1b_retrieved_chunks
    
    # Detect languages for each context
    lang_1 = detect_language(cp1b_1_content)
    lang_2 = detect_language(cp1b_2_content)
    lang_3 = detect_language(cp1b_3_content)
    
    # Helper function to choose appropriate chain
    def get_chain_for_language(language, standard_chain, multilingual_chain):
        return multilingual_chain if language != 'en' else standard_chain
    
    # Route to appropriate chains based on language
    chain_1 = get_chain_for_language(lang_1, cp1b_criterion_1_chain, cp1b_criterion_1_chain_noneng)
    chain_2 = get_chain_for_language(lang_2, cp1b_criterion_2_chain, cp1b_criterion_2_chain_noneng)
    chain_3 = get_chain_for_language(lang_3, cp1b_criterion_3_chain, cp1b_criterion_3_chain_noneng)
    
    # Execute evaluations
    cp1b_criterion_1_result = chain_1.invoke({"context": cp1b_1_content})
    cp1b_criterion_2_result = chain_2.invoke({"context": cp1b_2_content})
    cp1b_criterion_3_result = chain_3.invoke({"context": cp1b_3_content})
    
    if return_only_result:
        # Return minimal structure for YES/NO chain
        return {
            "criterion_1_result": cp1b_criterion_1_result.content,
            "criterion_2_result": cp1b_criterion_2_result.content,
            "criterion_3_result": cp1b_criterion_3_result.content
        }
    else:
        # Return full details for analysis
        return {
            "criterion_1_result": cp1b_criterion_1_result.content,
            "criterion_2_result": cp1b_criterion_2_result.content,
            "criterion_3_result": cp1b_criterion_3_result.content,
            "languages_detected": {
                "context_1": lang_1,
                "context_2": lang_2,
                "context_3": lang_3
            }
        }

# Full detailed assessment chain
cp1b_complete_assessment_chain_detailed = (
    RunnableLambda(lambda x: evaluate_all_criteria_multilingual_cp1b(x, return_only_result=False))
    | cp1b_final_assessment_prompt 
    | llm
)

# Simple YES/NO assessment chain
cp1b_complete_assessment_chain_simple = (
    RunnableLambda(lambda x: evaluate_all_criteria_multilingual_cp1b(x, return_only_result=True))
    | cp1b_final_assessment_prompt 
    | llm
    | RunnableLambda(extract_yes_no_result)
)

def run_cp1b_assessment(country=None, detailed=True, print_results=True):
    """
    Main function to run CP1b assessment with a single call
    
    Args:
        detailed: If True, returns full details; if False, returns only YES/NO
        print_results: If True, prints formatted results to console
    
    Returns:
        Assessment result (detailed dict or simple YES/NO string)
    """
    retrieved_chunks = get_chunks_for_cp1b(country)

    if detailed:
        result = cp1b_complete_assessment_chain_detailed.invoke(retrieved_chunks)
        if print_results:
            print("Detailed Assessment:")
            print(result.content)
        return result
    else:
        result = cp1b_complete_assessment_chain_simple.invoke(retrieved_chunks)
        if print_results:
            print("Simple Assessment:", result)
        return result