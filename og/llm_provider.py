"""
LLM Provider Abstraction Layer
Supports OpenAI, Google Gemini, and MMR fallback
"""

import time
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from openai import OpenAI
import google.generativeai as genai


# ============================================
# Prompt Templates
# ============================================

SYSTEM_PROMPT = """You are a precise policy assistant. Answer questions based ONLY on the provided document excerpts.

Rules:
1. Use only information from the context provided
2. If the answer isn't in the context, respond: "This information is not available in the provided context"
3. Write in clear, natural language:
   - Keep section titles and headings from the source (e.g., "2.1 Annual Performance Review:")
   - Preserve numbered/bulleted structure from source documents
   - Identify key policy terms (required, mandatory, eligible, must, approval, within X days)
   - Break information into logical paragraphs
   - Use one sentence per line for distinct pieces of information
4. Structure multi-part answers clearly:
   - Start with relevant section headings
   - Group related details together
   - Present lists as separate lines (one item per line)
5. Be concise - answer the specific question without unnecessary elaboration
6. Do not make assumptions or add external knowledge
7. Preserve the original document structure and terminology"""

USER_PROMPT_TEMPLATE = """Question: {query}

Context from documents:
{context}

Provide a clear, accurate answer using ONLY the information from the context above.

Guidelines:
- Keep section titles as they appear in the source (e.g., "2.1 Annual Performance Review")
- Identify and mention key policy terms (required, mandatory, eligible, must, within X days)
- If the answer has multiple parts, organize them logically
- Put each distinct piece of information on a new line
- If there's a list in the source, preserve it (one item per line)
- Keep the answer concise and directly address the question

Focus on accuracy and completeness. Do not add formatting markup - write plain text."""


# ============================================
# Abstract Base Class
# ============================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, config: Dict):
        self.config = config
        self.temperature = config.get('temperature', 0.3)
        self.max_tokens = config.get('max_tokens', 500)

    @abstractmethod
    def generate_answer(self, query: str, context: List[str]) -> Dict[str, Any]:
        """
        Generate answer from query and context

        Returns:
            {
                "answer": str,
                "provider": str,
                "tokens_used": int,
                "fallback_used": bool
            }
        """
        pass

    def format_context(self, context: List[str]) -> str:
        """Format context sentences into a single string"""
        return "\n\n".join([f"[{i+1}] {sentence}" for i, sentence in enumerate(context)])


# ============================================
# OpenAI Provider
# ============================================

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""

    def __init__(self, config: Dict):
        super().__init__(config)
        api_key = config.get('openai_api_key')
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=api_key)
        self.model = config.get('openai_model', 'gpt-4-turbo-preview')
        self.timeout = config.get('timeout', 10)
        self.max_retries = config.get('max_retries', 3)

    def generate_answer(self, query: str, context: List[str]) -> Dict[str, Any]:
        """Generate answer using OpenAI"""
        formatted_context = self.format_context(context)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            query=query,
            context=formatted_context
        )

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )

                return {
                    "answer": response.choices[0].message.content.strip(),
                    "provider": "openai",
                    "tokens_used": response.usage.total_tokens,
                    "fallback_used": False
                }

            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed, raise error
                    raise Exception(f"OpenAI API failed after {self.max_retries} attempts: {str(e)}")


# ============================================
# Gemini Provider
# ============================================

class GeminiProvider(LLMProvider):
    """Google Gemini provider"""

    def __init__(self, config: Dict):
        super().__init__(config)
        api_key = config.get('gemini_api_key')
        if not api_key:
            raise ValueError("Gemini API key is required")

        genai.configure(api_key=api_key)
        self.model_name = config.get('gemini_model', 'gemini-1.5-pro')
        self.model = genai.GenerativeModel(self.model_name)
        self.timeout = config.get('timeout', 10)
        self.max_retries = config.get('max_retries', 3)

    def generate_answer(self, query: str, context: List[str]) -> Dict[str, Any]:
        """Generate answer using Google Gemini"""
        formatted_context = self.format_context(context)

        # Gemini uses a single prompt with system instructions embedded
        full_prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(query=query, context=formatted_context)}"

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )

                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                    request_options={'timeout': self.timeout}
                )

                # Extract token usage if available
                tokens_used = 0
                if hasattr(response, 'usage_metadata'):
                    tokens_used = (
                        response.usage_metadata.prompt_token_count +
                        response.usage_metadata.candidates_token_count
                    )

                return {
                    "answer": response.text.strip(),
                    "provider": "gemini",
                    "tokens_used": tokens_used,
                    "fallback_used": False
                }

            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed, raise error
                    raise Exception(f"Gemini API failed after {self.max_retries} attempts: {str(e)}")


# ============================================
# MMR Provider (Fallback)
# ============================================

class MMRProvider(LLMProvider):
    """Fallback provider using MMR sentence selection (no LLM)"""

    def __init__(self, config: Dict):
        super().__init__(config)

    def generate_answer(self, query: str, context: List[str]) -> Dict[str, Any]:
        """Return concatenated context sentences (existing MMR behavior)"""
        # Simply join the retrieved sentences
        answer = " ".join(context)

        return {
            "answer": answer,
            "provider": "mmr",
            "tokens_used": 0,
            "fallback_used": True
        }


# ============================================
# Provider Factory
# ============================================

class LLMProviderFactory:
    """Factory to create appropriate LLM provider"""

    @staticmethod
    def create_provider(config: Dict) -> LLMProvider:
        """
        Create provider based on configuration

        Args:
            config: Dictionary containing:
                - provider: "openai", "gemini", or "none"
                - openai_api_key: OpenAI API key (if using OpenAI)
                - gemini_api_key: Gemini API key (if using Gemini)
                - Other model-specific settings

        Returns:
            Appropriate LLMProvider instance
        """
        provider_type = config.get('provider', 'none').lower()
        fallback_enabled = config.get('fallback_to_mmr', True)

        try:
            # Try to create requested provider
            if provider_type == 'openai' and config.get('openai_api_key'):
                return OpenAIProvider(config)

            elif provider_type == 'gemini' and config.get('gemini_api_key'):
                return GeminiProvider(config)

            else:
                # No provider specified or no API key
                return MMRProvider(config)

        except Exception as e:
            # Provider initialization failed
            if fallback_enabled:
                print(f"Warning: {provider_type} provider initialization failed ({str(e)}). Falling back to MMR.")
                return MMRProvider(config)
            else:
                raise


# ============================================
# Wrapper with Automatic Fallback
# ============================================

class RobustLLMProvider:
    """Wrapper that handles runtime errors with automatic fallback"""

    def __init__(self, primary_provider: LLMProvider, fallback_provider: Optional[LLMProvider] = None):
        self.primary = primary_provider
        self.fallback = fallback_provider or MMRProvider({})

    def generate_answer(self, query: str, context: List[str]) -> Dict[str, Any]:
        """Generate answer with automatic fallback on errors"""
        try:
            return self.primary.generate_answer(query, context)
        except Exception as e:
            print(f"Warning: Primary provider failed ({str(e)}). Using fallback.")
            result = self.fallback.generate_answer(query, context)
            result['fallback_used'] = True
            result['error'] = str(e)
            return result
