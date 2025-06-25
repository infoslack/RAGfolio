from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from openai import AsyncOpenAI
from app.services.document_retriever import DocumentRetriever
from app.services.prompt_manager import PromptManager
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Generic type for analysis results


class BaseAnalyzer(ABC, Generic[T]):
    """Base class for all analyzers"""

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        document_retriever: DocumentRetriever,
        prompt_manager: PromptManager,
        model: str,
    ):
        self.client = openai_client
        self.document_retriever = document_retriever
        self.prompt_manager = prompt_manager
        self.model = model

    @abstractmethod
    async def analyze(self, ticker: str) -> T:
        """Main analysis method to be implemented by subclasses"""
        pass

    async def _call_openai_structured(
        self,
        prompt_name: str,
        user_content: str,
        response_model: type[T],
        temperature: float = 0,
    ) -> T:
        """Helper method to call OpenAI with structured output"""
        try:
            system_prompt = self.prompt_manager.get_prompt(prompt_name)

            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format=response_model,
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            logger.error(f"OpenAI call failed for {prompt_name}: {str(e)}")
            raise
