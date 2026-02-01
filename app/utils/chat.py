"""
Custom Chat wrapper for OpenAI
Fixes compatibility issues with langchain-openai
"""
from typing import List, Optional, Any, Mapping
from openai import OpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, LLMResult, Generation
from langchain.llms.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from config.settings import settings


class SimpleChatOpenAI(BaseLLM):
    """Simple OpenAI chat wrapper that actually works"""

    model: str = "gpt-4"
    temperature: float = 0
    client: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model: str = "gpt-4", temperature: float = 0, **kwargs):
        super().__init__(**kwargs)
        # Create OpenAI client without custom httpx - let it use defaults
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "openai-chat"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters"""
        return {"model": self.model, "temperature": self.temperature}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Call OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stop=stop
        )
        return response.choices[0].message.content

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        """Generate responses for multiple prompts"""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def predict(self, text: str, **kwargs) -> str:
        """Predict method for compatibility"""
        return self._call(text, **kwargs)


class SimpleGeminiChat(BaseLLM):
    """Gemini chat wrapper for LangChain compatibility"""

    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0
    client: Any = None
    model_name: str = "gemini-2.0-flash-exp"

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model: str = "gemini-2.0-flash-exp", temperature: float = 0, **kwargs):
        super().__init__(**kwargs)
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        self.model_name = model
        self.client = genai.GenerativeModel(model)
        self.model = model
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "gemini-chat"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters"""
        return {"model": self.model_name, "temperature": self.temperature}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Call Gemini API"""
        generation_config = {
            "temperature": self.temperature,
            "stop_sequences": stop or []
        }
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        """Generate responses for multiple prompts"""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def predict(self, text: str, **kwargs) -> str:
        """Predict method for compatibility"""
        return self._call(text, **kwargs)


def get_chat_llm(model: str = None, temperature: float = 0) -> BaseLLM:
    """Factory function to get appropriate chat LLM based on provider.

    Args:
        model: Optional model name override
        temperature: Temperature for generation (0-1)

    Returns:
        SimpleGeminiChat if GEMINI_API_KEY is set, else SimpleChatOpenAI
    """
    if settings.llm_provider == "gemini":
        return SimpleGeminiChat(
            model=model or "gemini-pro",
            temperature=temperature
        )
    else:
        # OpenAI fallback (still available and intact)
        return SimpleChatOpenAI(
            model=model or "gpt-4",
            temperature=temperature
        )
