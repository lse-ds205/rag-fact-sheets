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
Etc...
"""