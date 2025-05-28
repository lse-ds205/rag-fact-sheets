
"""
LLM Response Generation Prompts
"""


LLM_SYSTEM_PROMPT = """You are an expert climate policy analyst. Always respond with valid JSON in the exact format requested."""

# Prompt for when JSON structure is enforced by the API (guided JSON)
LLM_GUIDED_PROMPT_TEMPLATE = """You are an expert climate policy analyst. Your task is to answer questions about climate policies based on provided document chunks from NDC (Nationally Determined Contributions) documents.

INSTRUCTIONS:
1. Include the original question in your response
2. Provide a comprehensive answer based ONLY on the information in the provided chunks
3. In citations, include ALL chunks that contributed to your answer
4. For each citation, explain how you used that specific chunk
5. Be precise and factual - do not add information not present in the chunks
6. If chunks contain conflicting information, acknowledge this in your response

{CONTEXT_CHUNKS}

QUESTION: {USER_QUESTION}

Please provide a structured response that includes the original question, your answer, citations, and metadata."""

# Prompt for when JSON structure must be explicitly instructed (fallback)
LLM_FALLBACK_PROMPT_TEMPLATE = """You are an expert climate policy analyst. Your task is to answer questions about climate policies based on provided document chunks from NDC (Nationally Determined Contributions) documents.

INSTRUCTIONS:
1. Include the original question in your response
2. Provide a comprehensive answer based ONLY on the information in the provided chunks
3. In citations, include ALL chunks that contributed to your answer
4. For each citation, explain how you used that specific chunk
5. Be precise and factual - do not add information not present in the chunks
6. If chunks contain conflicting information, acknowledge this in your response
7. CRITICAL: Respond ONLY with valid JSON in the exact format specified below

{CONTEXT_CHUNKS}

QUESTION: {USER_QUESTION}

REQUIRED JSON RESPONSE FORMAT (respond with JSON only, no other text):
{{
  "question": "{USER_QUESTION}",
  "answer": {{
    "summary": "Brief 2-3 sentence summary of the main answer",
    "detailed_response": "Comprehensive answer to the question with full context and analysis"
  }},
  "citations": [
    {{
      "id": <chunk_id>,
      "doc_id": "<doc_id>",
      "content": "<full_chunk_content>",
      "chunk_index": <chunk_index>,
      "paragraph": <paragraph_number_or_null>,
      "language": <language_or_null>,
      "chunk_metadata": <full_chunk_metadata_object>,
      "country": "<country>",
      "cos_similarity_score": <similarity_score>,
      "how_used": "Explanation of how this chunk contributed to the answer"
    }}
  ],
  "metadata": {{
    "chunks_cited": <number_of_chunks_cited>,
    "primary_countries": [<list_of_main_countries_discussed>]
  }}
}}

RESPONSE (JSON only):"""

# System prompt for fallback mode (more explicit about JSON requirements)
LLM_FALLBACK_SYSTEM_PROMPT = """You are an expert climate policy analyst. You must ALWAYS respond with valid JSON in the exact format requested. Never include any text outside the JSON structure. Never use markdown formatting or code blocks."""


# ------------------------------------------------------------------------------------------------

"""
These are the prompts for the question-answering pipeline.
"""


QUESTION_PROMPT_1 = """What does the country promise as their emissions reduction target?

Please extract:
- The specific reduction percentage or amount
- The target year(s) for achieving this reduction
- Whether the target is absolute or relative
- Whether it's an economy-wide target or sector-specific
- Any conditions attached to the target (conditional vs. unconditional)
"""

QUESTION_PROMPT_2 = """What year is the country using as baseline for their emissions reduction target?

Please determine:
- The baseline year explicitly stated in the document
- Whether they're reporting a business-as-usual (BAU) target rather than a base year target
- What sectors are covered by the target (e.g., Energy, Industrial Processes, Land use/LULUCF, etc.)
- What greenhouse gases are covered by the target (CO2, CH4, N2O, etc.)
- The emissions quantity in the baseline year (if reported)
- The projected emissions levels under the BAU scenario (if relevant)
"""

QUESTION_PROMPT_3 = """What promises or commitments in this version of the NDC are different from the previous version?

Please identify:
- Changes in emissions reduction targets (increased/decreased ambition)
- New sectors or gases that have been included
- Changes in baseline year or reference scenarios
- New policies or implementation strategies
- Changes in conditional vs. unconditional commitments
- Any explicit statements comparing current NDC to previous versions
"""

QUESTION_PROMPT_4 = """What specific policies or strategies does the country propose to meet its climate targets?

Please analyze:
- Whether they breakdown their target by sector
- If they quantify expected emissions reductions from different policies
- Whether they plan to use international carbon markets to meet targets
- If so, how many carbon credits they plan on purchasing
- Major policy instruments mentioned (e.g., carbon pricing, regulations, subsidies)
- Implementation timelines for key policies
"""

QUESTION_PROMPT_5 = """Do they specify which sectors of their economy will be the hardest to reduce emissions in?

Please identify:
- Explicitly mentioned challenging sectors
- Sectors requiring international support or technology transfer
- Sectors with limited mitigation potential
- Sectors with highest projected emissions growth
- Any quantitative analysis of sectoral challenges
"""

QUESTION_PROMPT_6 = """What adaptation measures does the country propose to implement?

Please extract:
- Key adaptation priorities by sector (water, agriculture, health, etc.)
- Specific adaptation projects or programs mentioned
- Estimated costs of adaptation measures
- Funding sources identified for adaptation
- Institutional arrangements for adaptation planning
"""

QUESTION_PROMPT_7 = """What climate finance needs does the country identify in its NDC?

Please analyze:
- Total financial support requested (if specified)
- Breakdown between mitigation and adaptation financing
- Domestic vs. international financing sources
- Specific funding mechanisms mentioned
- Project-level or program-level financial estimates
- Any innovative financing approaches proposed
"""

QUESTION_PROMPT_8 = """How does the country address climate justice, equity, or fair share considerations in its NDC?

Please identify:
- References to common but differentiated responsibilities
- Discussions of historical emissions or responsibility
- Incorporation of social justice, gender, or indigenous perspectives
- Mentions of just transition for workers in high-emission industries
- Discussion of equitable burden-sharing among countries
"""