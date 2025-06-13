from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# For the template generating LLM
template_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation. 
    Your task is to create a template for the structure of a 1 page report on {topic}. 
     
     Here are some example templates to base your response on:
        ##METADATA 

        *Country: 

        ##REPORT 

        Q1: What are the relevant background information for indicator X, Y or Z? 
        … 
        Question N (CHALLENGING): How has the legislation of this country changed in the past 50 years? 
        
     
        ##TOPIC: Just transition 

        *Countries with relevant information: … 

        ##REPORT 

        Q1: Which countries have made plans for a just energetic transition? 

        A: Country A [citation], B [citation] and C [citation] have a similar law. They all promise X, Y and Z. On the other hand, the following countries … differ because … 
        A: … 
        
        A law meets this criterion if it includes a clear statement to meet the goals of the Paris Agreement 
        OR a national long-term decarbonisation target.

    Respond with only the template of the {topic} report and nothing else."""),
    ("human", "Context: {topic}")
])

# Seperate template into sub-sections USE LOW POWER LMMM
section_seperator_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation.
    Your task is to extract the subsections from {template}.         
    Respond with only the template subsections nothing else."""),
# SPECIFY HOW TO STRUCUTRE SUBSECTIONS SO CAN BE EXTRACTED EASILY
    ("human", "{template}")
])

# For the hypothetical response generating LLM
hypothetical_response_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation. 
    Your task is to generate a hypothetical response for the following question: {subsection}. 
    The response should be based on the template {template}, a template for a report on topic {topic}.
    Respond with only the hypothetical response nothing else."""),
    ("human", "{subsection}, {template}, {topic}")
])
# Prompts for each of the sub-section models
subsection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation. 
    Your task is to create a one paragraph {subsection} as part of a report on {topic}.
    Use only information on {context} to create the paragraph.
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).
    The paragraph should be concise and informative, summarizing the key points relevant to the subsection.     
    Respond with only the paragraph for that subsections nothing else."""),
    ("human", "{subsection}, {context}, {topic}")
])


# THIS COULD BE DONE WITHOUT USING AN LLM???
# Prompt for the compling model
compiling_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation. 
    Your task is to compile the following subsections {all_subsections}.
    The report should be structured according to the template {template}.
    The report must match exactly the template structure.
    Respond with only the compiled report nothing else."""),
    ("human", "{all_subsections}, {template}")
])

# Prompt for the checking model
checking_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation. 
    Your task is to check the following subsections {all_subsections} for consistency and completeness.
    Also ensure that the report {report} matches the template {template}.
    Also ensure that the report is related to the topic {topic}.
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).
    If 
        1) One of the subsections appears incorrect or incomplete → output ONLY: (**subsection name**), incomplete
        2) The report does not match the template → output ONLY: not match
        3) The report is not related to the topic → output ONLY: not related
        4) Everything is correct → output ONLY: ok
     
    Respond with only the result of the check (ok, not related, not match, (**subsection name**) incomplete) nothing else.
    """),
    ("human", "{all_subsections}, {template}, {report}, {topic}, {context}")
])

# Prompt for the rewrite subsection model
rewrite_subsection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation. 
    Your task is to rewrite the following subsection {subsection} to make it more complete and consistent.
    Use the template {template} as a guide for the structure.
    The rewritten subsection should be concise and informative, summarizing the key points relevant to the subsection.
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).
    Respond with only the rewritten subsection nothing else."""),
    ("human", "{subsection}, {template}, {context}")
])