from pathlib import Path
import sys
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import os
from openai import OpenAI

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from schemas.llm import LLMResponseModel
from constants.prompts import (
    LLM_SYSTEM_PROMPT,
    LLM_GUIDED_PROMPT_TEMPLATE,
    LLM_FALLBACK_PROMPT_TEMPLATE,
    LLM_FALLBACK_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)


class ChunkFormatter:
    """
    Format chunks for LLM consumption.
    """
    
    @staticmethod
    def format_chunks_for_context(chunks: List[Dict[str, Any]]) -> str:
        """
        Format chunks into a readable context string for the LLM.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No context chunks provided."
        
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"""
                CHUNK {i}:
                ID: {chunk.get('id', 'N/A')}
                Document: {chunk.get('doc_id', 'N/A')}
                Country: {chunk.get('country', 'N/A')}
                Content: {chunk.get('content', 'N/A')}
                Similarity Score: {chunk.get('cos_similarity_score', chunk.get('similarity_score', 'N/A'))}
                ---"""
            formatted_chunks.append(chunk_text)
        
        context_header = f"CONTEXT INFORMATION ({len(chunks)} chunks):\n"
        return context_header + "\n".join(formatted_chunks)


class LLMClient:
    """
    Handle LLM API interactions using OpenAI library with custom endpoint.
    """
    
    def __init__(self, supports_guided_json: bool = True):
        self.supports_guided_json = supports_guided_json
        self.client = None
        
        # Load LLM configuration from environment variables
        self.model = os.getenv('LLM_MODEL', 'meta-llama/Meta-Llama-3.1-70B-Instruct')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '4000'))
        
        # Initialize the OpenAI client with AI API endpoint
        try:
            api_key = os.getenv('AI_API_KEY')
            base_url = os.getenv('AI_BASE_URL')
            
            if not api_key:
                logger.error("AI_API_KEY not found in environment variables")
                return
                
            if not base_url:
                logger.error("AI_BASE_URL not found in environment variables")
                return
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info(f"Successfully initialized LLM client")
            logger.info(f"Model: {self.model}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
            logger.info(f"Guided JSON: {self.supports_guided_json}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.client = None
    
    def create_llm_prompt(self, question: str, formatted_chunks: str) -> str:
        """
        Create a structured prompt for the LLM based on guided JSON support.
        
        Args:
            question: User's question
            formatted_chunks: Formatted context chunks
            
        Returns:
            Complete prompt for the LLM
        """
        if self.supports_guided_json:
            # Use clean prompt for guided JSON
            prompt = LLM_GUIDED_PROMPT_TEMPLATE.format(
                CONTEXT_CHUNKS=formatted_chunks,
                USER_QUESTION=question
            )
        else:
            # Use explicit JSON instruction prompt for fallback
            prompt = LLM_FALLBACK_PROMPT_TEMPLATE.format(
                CONTEXT_CHUNKS=formatted_chunks,
                USER_QUESTION=question
            )
        
        return prompt
    
    def call_llm(self, prompt: str) -> LLMResponseModel:
        """
        Make API call to LLM service with guided JSON response or fallback parsing.
        Uses configuration from environment variables.
        
        Args:
            prompt: The complete prompt to send
            
        Returns:
            Structured LLMResponseModel object
        """
        if not self.client:
            logger.error("LLM client not initialized")
            return None
        
        try:
            logger.info(f"Making LLM API call with model: {self.model} (guided_json: {self.supports_guided_json})")
            logger.debug(f"Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
            
            # Determine system prompt based on guided JSON support
            system_prompt = LLM_SYSTEM_PROMPT if self.supports_guided_json else LLM_FALLBACK_SYSTEM_PROMPT
            
            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Add guided JSON if supported
            if self.supports_guided_json:
                api_params["extra_body"] = {
                    "guided_json": LLMResponseModel.model_json_schema()
                }
            
            # Make API call
            response = self.client.chat.completions.create(**api_params)
            response_content = response.choices[0].message.content
            
            # Parse response based on guided JSON support
            if self.supports_guided_json:
                # Direct parsing since JSON structure is enforced
                parsed_response = LLMResponseModel.model_validate_json(response_content)
            else:
                # Fallback parsing with error handling
                parsed_response = self._parse_fallback_response(response_content)
            
            logger.info("Successfully received and parsed LLM response")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return None
    
    def _parse_fallback_response(self, response_content: str) -> LLMResponseModel:
        """
        Parse LLM response when guided JSON is not available.
        
        Args:
            response_content: Raw response text from LLM
            
        Returns:
            Parsed LLMResponseModel object
        """
        try:
            # Extract JSON from response (in case there's extra text)
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_content[json_start:json_end]
            
            # Parse and validate with Pydantic
            parsed_response = LLMResponseModel.model_validate_json(json_text)
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error parsing fallback response: {e}")
            raise


class ResponseProcessor:
    """
    Process and validate LLM responses.
    """
    
    @staticmethod
    def process_llm_response(llm_response: LLMResponseModel, original_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process and validate the structured LLM response.
        
        Args:
            llm_response: Structured LLMResponseModel from the API
            original_chunks: Original chunk data for validation
            
        Returns:
            Processed response dictionary
        """
        try:
            if not llm_response:
                logger.error("No response from LLM")
                return ResponseProcessor._create_error_response("No response from LLM")
            
            # Convert Pydantic model to dictionary
            response_dict = llm_response.model_dump()
            
            # Validate and enrich citations with original chunk data
            validated_response = ResponseProcessor._validate_citations(response_dict, original_chunks)
            
            logger.info("Successfully processed structured LLM response")
            return validated_response
            
        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
            return ResponseProcessor._create_error_response(f"Processing error: {e}")
    
    @staticmethod
    def _validate_citations(response_data: Dict[str, Any], original_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and enrich citations with original chunk data."""
        try:
            # Create lookup dict for original chunks
            chunk_lookup = {chunk['id']: chunk for chunk in original_chunks}
            
            validated_citations = []
            
            for citation in response_data['citations']:
                chunk_id = citation.get('id')
                
                if chunk_id in chunk_lookup:
                    # Use original chunk data and add how_used from LLM
                    original_chunk = chunk_lookup[chunk_id].copy()
                    
                    # Rename similarity_score to cos_similarity_score if needed
                    if 'similarity_score' in original_chunk:
                        original_chunk['cos_similarity_score'] = original_chunk.pop('similarity_score')
                    
                    # Add how_used from LLM response
                    original_chunk['how_used'] = citation.get('how_used', 'No explanation provided')
                    
                    validated_citations.append(original_chunk)
                else:
                    logger.warning(f"Citation references unknown chunk ID: {chunk_id}")
                    # Keep the citation as-is even if we can't validate it
                    validated_citations.append(citation)
            
            # Update response with validated citations
            response_data['citations'] = validated_citations
            response_data['metadata']['chunks_cited'] = len(validated_citations)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error validating citations: {e}")
            return response_data
    
    @staticmethod
    def _create_error_response(error_message: str, question: str = "") -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "question": question,
            "answer": {
                "summary": "Error processing request",
                "detailed_response": f"An error occurred while processing the request: {error_message}"
            },
            "citations": [],
            "metadata": {
                "chunks_cited": 0,
                "primary_countries": []
            },
            "error": error_message
        }


class ConfidenceClassification:
    """
    Classifies retrieval results into confidence bands based on various scores.
    Used to provide quality indicators for LLM responses.
    """
    
    def __init__(self):
        """Initialize confidence thresholds for classification"""
        # Configurable thresholds for confidence bands
        self.thresholds = {
            "high": {
                "combined_score": 0.75,
                "similarity_score": 0.8,
                "regex_score": 0.7,
                "fuzzy_score": 0.7
            },
            "average": {
                "combined_score": 0.5,
                "similarity_score": 0.6,
                "regex_score": 0.4,
                "fuzzy_score": 0.4
            }
            # Below average thresholds = Low confidence
        }
    
    def classify_response(self, llm_response: Dict[str, Any], retrieve_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an LLM response based on retrieval scores.
        
        Args:
            llm_response: The response from the LLM
            retrieve_data: The retrieval data used to generate the response
            
        Returns:
            Updated response with confidence band added
        """
        # Deep copy to avoid modifying original
        result = llm_response.copy()
        
        # Process each question in the response
        for q_key, q_data in result.get("questions", {}).items():
            # Get corresponding retrieval data
            retrieve_q_data = retrieve_data.get("questions", {}).get(q_key, {})
            
            # Calculate confidence band based on retrieval scores
            confidence_band = self._calculate_confidence_band(retrieve_q_data)
            
            # Add confidence band to response
            q_data["confidence_band"] = confidence_band
        
        return result
    
    def _calculate_confidence_band(self, question_data: Dict[str, Any]) -> str:
        """
        Calculate confidence band based on retrieval scores.
        
        Args:
            question_data: Question data from retrieval
            
        Returns:
            Confidence band: "High", "Average", or "Low"
        """
        # Extract scores from both traditional and hoprag methods
        scores = self._extract_scores(question_data)
        
        if not scores or not scores.get("chunks", []):
            return "Low"  # No scores available
        
        # Calculate average scores across top chunks (up to 5)
        top_chunks = scores.get("chunks", [])[:5]
        
        avg_scores = {
            "combined_score": sum(c.get("combined_score", 0) for c in top_chunks) / len(top_chunks) if top_chunks else 0,
            "similarity_score": sum(c.get("similarity_score", c.get("cos_similarity_score", 0)) for c in top_chunks) / len(top_chunks) if top_chunks else 0,
            "regex_score": sum(c.get("regex_score", 0) for c in top_chunks) / len(top_chunks) if top_chunks else 0,
            "fuzzy_score": sum(c.get("fuzzy_score", 0) for c in top_chunks) / len(top_chunks) if top_chunks else 0
        }
        
        # Check if scores meet high confidence thresholds
        if (avg_scores["combined_score"] >= self.thresholds["high"]["combined_score"] and 
            (avg_scores["similarity_score"] >= self.thresholds["high"]["similarity_score"] or 
             avg_scores["regex_score"] >= self.thresholds["high"]["regex_score"] or
             avg_scores["fuzzy_score"] >= self.thresholds["high"]["fuzzy_score"])):
            return "High"
        
        # Check if scores meet average confidence thresholds
        if (avg_scores["combined_score"] >= self.thresholds["average"]["combined_score"] and 
            (avg_scores["similarity_score"] >= self.thresholds["average"]["similarity_score"] or 
             avg_scores["regex_score"] >= self.thresholds["average"]["regex_score"] or
             avg_scores["fuzzy_score"] >= self.thresholds["average"]["fuzzy_score"])):
            return "Average"
        
        # Default to low confidence
        return "Low"
    
    def _extract_scores(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract scores from retrieval data.
        
        Args:
            question_data: Question data from retrieval
            
        Returns:
            Dictionary with score information
        """
        result = {"chunks": []}
        
        # Combine chunks from traditional and hoprag retrieval methods
        for method in ["traditional", "hoprag"]:
            method_data = question_data.get("retrieval_methods", {}).get(method, {})
            chunks = method_data.get("results", {}).get("evaluated_chunks", [])
            
            for chunk in chunks:
                # Some chunks might have different score naming conventions
                chunk_data = {
                    "combined_score": chunk.get("combined_score", 0),
                    "similarity_score": chunk.get("similarity_score", chunk.get("cos_similarity_score", 0)),
                    "regex_score": chunk.get("regex_score", 0),
                    "fuzzy_score": chunk.get("fuzzy_score", 0)
                }
                
                result["chunks"].append(chunk_data)
        
        # Sort by combined score
        result["chunks"].sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        return result