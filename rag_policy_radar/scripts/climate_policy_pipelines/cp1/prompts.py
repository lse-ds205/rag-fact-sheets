from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate



# Define individual criterion evaluation prompts
cp1a_criterion_1_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation. 
    Your task is to evaluate whether a climate law sets a strategic direction for decarbonisation.
    
    A law meets this criterion if it includes a clear statement to meet the goals of the Paris Agreement 
    OR a national long-term decarbonisation target.
    
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).

     
    Respond with only 'YES' or 'NO' followed by a brief explanation."""),
    ("human", "Context: {context}\n\nDoes this law set a strategic direction for decarbonisation?")
])

cp1a_criterion_2_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation.
    Your task is to evaluate whether a climate law is enshrined in law.
    
    A law meets this criterion if it is legislative rather than executive 
    (except in particular political systems where executive action has legal force).
     
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).

    
    Respond with only 'YES' or 'NO' followed by a brief explanation."""),
    ("human", "Context: {context}\n\nIs this law enshrined in law (legislative rather than executive)?")
])

cp1a_criterion_3_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation.
    Your task is to evaluate whether a climate law sets out specific obligations.
    
    A law meets this criterion if it sets out at least one of the following:
    - Meeting a national target
    - Developing, revising, implementing or complying with domestic plans, strategies or policies
    - Developing policy instruments such as regulation, taxation or public spending in support of climate goals
    
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).

    Respond with only 'YES' or 'NO' followed by a brief explanation."""),
    ("human", "Context: {context}\n\nDoes this law set out the required obligations?")
])

cp1a_criterion_4_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation.
    Your task is to evaluate the exceptional case criterion.
    
    This criterion is met if there is a combination of:
    - A broad environmental law AND
    - A clearly linked executive climate strategy
    
    This combination may be sufficient to meet the framework criteria in exceptional cases.
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).

    
    Respond with only 'YES' or 'NO' followed by a brief explanation."""),
    ("human", "Context: {context}\n\nDoes this represent a valid exceptional case (broad environmental law + linked executive strategy)?")
])

# Final assessment prompt
cp1a_final_assessment_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst making a final assessment of climate legislation.
    
    A country is assessed as 'YES' for having framework climate law if:
    - Criteria 1, 2, AND 3 are all satisfied, OR
    - Criterion 4 is satisfied (exceptional case)
   
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).

    Based on the individual assessments, provide a final 'YES' or 'NO' answer with reasoning."""),
    ("human", """Individual criterion assessments:
    Criterion 1 (Strategic direction): {criterion_1_result}
    Criterion 2 (Enshrined in law): {criterion_2_result}
    Criterion 3 (Obligations): {criterion_3_result}
    Criterion 4 (Exceptional case): {criterion_4_result}
    
    What is the final assessment?""")
])

#CP1a large context prompt

# Single comprehensive climate legislation assessment prompt
comprehensive_assessment_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation assessment. 

Your task is to evaluate whether a country has a framework climate law based on specific criteria and provide a structured markdown assessment.

EVALUATION CRITERIA:
A country is assessed as 'YES' if it has a framework climate law that fulfils ALL of criteria 1, 2, AND 3, OR criterion 4:

1. STRATEGIC DIRECTION: Sets a strategic direction for decarbonisation (must include a clear statement to meet the goals of the Paris Agreement OR a national long-term decarbonisation target)

2. ENSHRINED IN LAW: Is enshrined in law (must be legislative rather than executive, except in particular political systems)

3. OBLIGATIONS: Sets out at least one of the following obligations:
   - Meeting a national target
   - Developing, revising, implementing or complying with domestic plans, strategies or policies
   - Developing policy instruments such as regulation, taxation or public spending in support of climate goals

4. EXCEPTIONAL CASE: The combination of a broad environmental law AND a clearly linked executive climate strategy may be sufficient to meet these criteria

ASSESSMENT LOGIC:
- If criteria 1, 2, AND 3 are all satisfied → YES
- If criterion 4 is satisfied → YES
- Otherwise → NO

OUTPUT FORMAT:
Provide your assessment in the following markdown format:

```markdown
# Climate Legislation Assessment: CP 1.a Framework Climate Law

## Individual Criterion Evaluation

### Criterion 1: Strategic Direction for Decarbonisation
**Result:** [YES/NO]
**Reasoning:** [Brief explanation of whether the law includes clear Paris Agreement goals or long-term decarbonisation targets]

### Criterion 2: Enshrined in Law
**Result:** [YES/NO]
**Reasoning:** [Brief explanation of whether this is legislative rather than executive]

### Criterion 3: Sets Out Obligations
**Result:** [YES/NO]
**Reasoning:** [Brief explanation of which obligations are present, if any]

### Criterion 4: Exceptional Case
**Result:** [YES/NO]
**Reasoning:** [Brief explanation of whether broad environmental law + executive strategy combination exists]

## Final Assessment

**Overall Result:** [YES/NO]

**Logic Applied:** [Explain whether criteria 1+2+3 are satisfied OR criterion 4 is satisfied]

**Conclusion:** [Brief summary of why the country does/does not have a framework climate law]
```"""),
    ("human", "Context: {context}\n\nPlease evaluate whether this country has a framework climate law based on the provided context. For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).")
])

# CP1b prompts  

# Define individual criterion evaluation prompts
cp1b_criterion_1_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation. 
    Your task is to evaluate whether a country's framework climate **Specification of who is accountable to whom for at least one stated obligation 
    (e.g. accountability of executive to parliament, or private parties to executive authorities)**
    
    These are examples of the types of relationships that usually qualify as having specification of who is accountable to whom:
        a.	executive to parliament 
        b.	executive to executive and/or administrative agencies 
        c.	national to sub-national and sub-national to national
        d.	executive to judiciary
        e.	executive to expert bodies
        f.	executive to citizens 
        g.	private parties to citizens 
        h.	private parties to executive authorities.
    
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).

    Respond with only 'YES' or 'NO' followed by a brief explanation."""),
    ("human", "Context: {context}\n\nDoes this country's framework climate law specify who is accountable to whom for at least one stated obligation?")
])

cp1b_criterion_2_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation.
    Your task is to evaluate whether a country's framework climate law contains the following accountability element:
    **specification of how compliance is assessed for at least one stated obligation 
    (e.g. transparency mechanisms in the form of monitoring, reporting and verification, parliamentary oversight, expert assessments, court proceedings)**
        
    Some of the most common ways of assessing compliance are:
    a.	transparency mechanisms in the form of monitoring, reporting and verification (MRV). Although this type of mechanism does not specify a decision-maker responsible for assessing whether an obligation has been met, this type of process is crucial for ensuring that there is political accountability for the implementation of climate law
    b.	parliamentary oversight, which can also be used to assess the effectiveness of the legislation in achieving the stated aims
    c.	Expert assessment
    d.	Court proceedings
    e.	Alternative Dispute Resolution.
    
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).

    Respond with only 'YES' or 'NO' followed by a brief explanation."""),
    ("human", "Context: {context}\n\nIs this country's framework climate law specification of how compliance is assessed for at least one stated obligation?")
])

cp1b_criterion_3_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst specializing in climate legislation.
    Your task is to evaluate whether a country's framework climate law contains the following accountability element:
        **Specification of what happens in the case of non-compliance for at least one stated obligation (e.g. parliamentary intervention, judicial orders, financial penalties).**
        
    It is relatively rare for countries to specify what happens in the case of non-compliance. Relevant practices identified in the laws can be grouped under certain categories. Some examples are:
    a.	Parliamentary intervention
    b.	Governmental or ministerial intervention
    c.	Judicial orders
    d.	Orders and fines by regulators
    e.	Court imposed financial penalties
    f.	Regulatory financial penalties
    g.	Multi-stakeholder agreements.
    Note that if a country has multiple laws assessed under the previous indicator, all are considered under this indicator on whether they together contain the three accountability elements above.
    
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).

    Respond with only 'YES' or 'NO' followed by a brief explanation."""),
    ("human", "Context: {context}\n\nDoes this country's framework climate law specify what happens in the case of non-compliance for at least one stated obligation?")
])

# Final assessment prompt
cp1b_final_assessment_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal analyst making a final assessment of climate legislation.
    
    A country is assessed as 'YES' if the country's framework climate law specifies ALL THREE of the following key accountability elements:
        Criterion 1: Specification of who is accountable to whom for at least one stated obligation
        Criterion 2: Specification of how compliance is assessed for at least one stated obligation
        Criterion 3: Specification of what happens in the case of non-compliance for at least one stated obligation


    If YES to all criteria, provide a final 'YES' or 'NO' answer with reasoning."""),
    ("human", """Individual criterion assessments:
    Criterion 1 (who is accountable to whom): {criterion_1_result}
    Criterion 2 (compliance is assessed): {criterion_2_result}
    Criterion 3 (case of non-compliance): {criterion_3_result}
    
    For any claims you make, you **MUST** include the page number and document citation in the format (page X, doc Y).

    What is the final assessment?""")
])