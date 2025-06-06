
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import BaseOutputParser

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

from scripts.climate_policy_pipelines.shared.llm_models import (
    llm,
    multilingual_llm,
    large_context_llm  
)

# CPIa Chains

# Standard chains
cp1a_criterion_1_chain = cp1a_criterion_1_prompt | llm
cp1a_criterion_2_chain = cp1a_criterion_2_prompt | llm
cp1a_criterion_3_chain = cp1a_criterion_3_prompt | llm
cp1a_criterion_4_chain = cp1a_criterion_4_prompt | llm

# Multilingual chains
cp1a_criterion_1_chain_noneng = cp1a_criterion_1_prompt | multilingual_llm
cp1a_criterion_2_chain_noneng = cp1a_criterion_2_prompt | multilingual_llm
cp1a_criterion_3_chain_noneng = cp1a_criterion_3_prompt | multilingual_llm
cp1a_criterion_4_chain_noneng = cp1a_criterion_4_prompt | multilingual_llm


# large context LLM chain
single_model_assessment_chain = comprehensive_assessment_prompt | large_context_llm


# CP1b assessment chains

# Standard chains
cp1b_criterion_1_chain = cp1b_criterion_1_prompt | llm
cp1b_criterion_2_chain = cp1b_criterion_2_prompt | llm
cp1b_criterion_3_chain = cp1b_criterion_3_prompt | llm

# Multilingual chains
cp1b_criterion_1_chain_noneng = cp1b_criterion_1_prompt | multilingual_llm
cp1b_criterion_2_chain_noneng = cp1b_criterion_2_prompt | multilingual_llm
cp1b_criterion_3_chain_noneng = cp1b_criterion_3_prompt | multilingual_llm