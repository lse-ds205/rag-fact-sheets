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
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema import BaseOutputParser
import re

load_dotenv()

from scripts.climate_policy_pipelines.shared.llm_models import (
    super_basic_model,
    standard_model,
    large_context_llm
)

from scripts.climate_policy_pipelines.report_generation.prompts import (
    template_prompt,
    section_seperator_prompt,  
    hypothetical_response_prompt,
    subsection_prompt,
    compiling_prompt,
    checking_prompt,
    rewrite_subsection_prompt
)

from scripts.retrieval.retrieval_pipeline import do_retrieval


class ReportWorkflow:
    def __init__(self, low_model, med_model, high_model):
        self.low_model = low_model
        self.med_model = med_model
        self.high_model = high_model

       
    def _extract_subsections(self, template):
        """Extract subsections from template - can be done without LLM"""
        # Simple regex-based extraction (replace with LLM if needed)
        sections = re.findall(r'Q\d+:.*?(?=Q\d+:|$)', template, re.DOTALL)
        return [section.strip() for section in sections if section.strip()]
   
    def _human_approval(self, content, prompt_msg):
        """Simulate human approval - replace with actual UI"""
        print(f"\n{prompt_msg}")
        print(content)
        return input("Approve? (y/n): ").lower() == 'y'
   
    def _retrieve_context(self, hypothetical_response, k=10):
        """Retrieve relevant context using hypothetical response"""
       
        try:
            # Use your custom retrieval function
            chunks = do_retrieval(hypothetical_response, k=k)
           
            # Format the chunks into a readable context string
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
       
        except Exception as e:
            print(f"Error in custom retrieval: {e}")
            return f"Error retrieving context: {str(e)}"
   
    def generate_template(self, topic):
        """Step 1-2: Generate and approve template"""
        chain = template_prompt | self.med_model
        template = chain.invoke({"topic": topic}).content
       
        if not self._human_approval(template, "Review template:"):
            # Retry with higher power model
            chain = template_prompt | self.high_model
            template = chain.invoke({"topic": topic}).content
           
        return template
   
    def extract_subsections(self, template):
        """Step 3: Extract subsections"""
        # Using simple extraction - uncomment below for LLM approach
        return self._extract_subsections(template)
       
        # LLM approach:
        # chain = section_seperator_prompt | self.low_model
        # result = chain.invoke({"template": template}).content
        # return result.split('\n')
   
    def generate_subsection_content(self, subsection, template, topic):
        """Step 4-6: Generate hypothetical response, retrieve context, create actual subsection"""
        # Generate hypothetical response
        hyp_chain = hypothetical_response_prompt | self.med_model
        hypothetical = hyp_chain.invoke({
            "subsection": subsection,
            "template": template,
            "topic": topic
        }).content
       
        # Retrieve context
        context = self._retrieve_context(hypothetical)
       
        # Generate actual subsection
        sub_chain = subsection_prompt | self.high_model
        content = sub_chain.invoke({
            "subsection": subsection,
            "context": context,
            "topic": topic
        }).content

        return content, context
          
    def compile_report(self, subsections, template):
        """Step 7: Compile subsections into report"""
        chain = compiling_prompt | self.low_model
        return chain.invoke({
            "all_subsections": "\n\n".join(subsections),
            "template": template
        }).content
   
    def quality_check_and_fix(self, subsections, template, report, topic, contexts=None):
        """Step 8: Quality check with potential fixes"""
        check_chain = checking_prompt | self.high_model

        if contexts:
            combined_context = "\n\n".join(contexts)
        else:
            # Fallback: generate new context from the report content
            combined_context = self._retrieve_context(report)
       
        max_retries = 3
        for attempt in range(max_retries):
            check_result = check_chain.invoke({
                "all_subsections": "\n\n".join(subsections),
                "template": template,
                "report": report,
                "topic": topic,
                "context": combined_context
            }).content.strip()
           
            if check_result == "ok":
                return report, subsections
           
            # Handle different failure cases
            if "incomplete" in check_result:
                # Extract subsection name and rewrite
                section_match = re.search(r'\*\*(.*?)\*\*', check_result)
                if section_match:
                    section_name = section_match.group(1)
                    # Find and rewrite the problematic subsection
                    for i, subsection in enumerate(subsections):
                        if section_name.lower() in subsection.lower():
                            rewrite_chain = rewrite_subsection_prompt | self.high_model
                            subsections[i] = rewrite_chain.invoke({
                                "subsection": subsection,
                                "template": template,
                                "context": combined_context  # Reuse same context

                            }).content
                            break
           
            elif check_result in ["not match", "not related"]:
                # Regenerate entire report
                report = self.compile_report(subsections, template)
       
        return report, subsections
   
    def run_workflow(self, topic):
        """Main workflow execution"""
        print(f"Starting report generation for topic: {topic}")
       
        # Step 1-2: Generate and approve template
        template = self.generate_template(topic)
       
        # Step 3: Extract subsections
        subsections_list = self.extract_subsections(template)
       
        # Step 4-6: Generate content for each subsection
        subsection_contents = []
        contexts = []  # Store contexts for reuse
        for subsection in subsections_list:
            content, context = self.generate_subsection_content(subsection, template, topic)
            subsection_contents.append(content)
            contexts.append(context)
       
        # Step 7: Compile report
        report = self.compile_report(subsection_contents, template)
       
        # Step 8: Quality check and fix (pass contexts)
        final_report, final_subsections = self.quality_check_and_fix(
            subsection_contents, template, report, topic, contexts
        )
       
        # Step 9: Human final approval
        if self._human_approval(final_report, "Final report review:"):
            print("Report approved and completed!")
            return final_report
        else:
            feedback = input("Please provide feedback: ")
            print(f"Report rejected. Feedback: {feedback}")
            return None

# Usage
report_workflow = ReportWorkflow(
    low_model=super_basic_model,
    med_model=standard_model,
    high_model=large_context_llm
)