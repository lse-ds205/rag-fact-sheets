"""
These are pipeline-level prompts - they are the major guiding prompts
"""

PIPELINE_PROMPT_1 = """
You are a helpful assistant.
Based on:
{ADDING_BOOSTER_PROMPT_1_FIRST}

And answers from these questions:
Question 1: {ANSWER_FROM_CHUNK_PROMPT_2}
Question 2: {ANSWER_FROM_CHUNK_PROMPT_3}

Tell me the answer from user's question. In this format:
<format> etc...
...
etc.

Now, answer this:
{USER_QUESTION}

Output:
"""

PIPELINE_PROMPT_2 = """
Placeholder Pipeline Prompt 2
"""

PIPELINE_PROMPT_3 = """
Placeholder Pipeline Prompt 3
"""

# ------------------------------------------------------------------------------------------------

"""
These are booster-level prompts - they are the minor guiding prompts, but ultimately feed into the pipeline prompts
"""

BOOSTER_PROMPT_1 = """
Placeholder Booster Prompt 1
Optional: {CHUNK_PROMPT_1}
"""

BOOSTER_PROMPT_2 = """
Placeholder Booster Prompt 2
"""

BOOSTER_PROMPT_3 = """
Placeholder Booster Prompt 3
"""

# ------------------------------------------------------------------------------------------------

"""
These are chunk-level prompts - they are the minor guiding prompts, but ultimately feed into the pipeline prompts to provide greater context
"""

CHUNK_PROMPT_1 = "Example: is this chunk about climate? Return only one word, 'yes' or 'no'"
CHUNK_PROMPT_2 = "Example: is this chunk about politics? Return only one word, 'yes' or 'no'"
CHUNK_PROMPT_3 = "Example: is this chunk about science? Return only one word, 'yes' or 'no'"

# ------------------------------------------------------------------------------------------------

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
Etc...
"""