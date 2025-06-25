from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Optional
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
        config_loader,
        model: str,
        temperature: float = 0.0,
        document_limit: int = 5,
    ):
        self.client = openai_client
        self.document_retriever = document_retriever
        self.prompt_manager = prompt_manager
        self.config_loader = config_loader
        self.model = model
        self.temperature = temperature
        self.document_limit = document_limit

    @abstractmethod
    async def analyze(self, ticker: str) -> T:
        """Main analysis method to be implemented by subclasses"""
        pass

    async def _call_openai_structured(
        self,
        prompt_name: str,
        user_content: str,
        response_model: Type[T],
        temperature: Optional[float] = None,
    ) -> T:
        """Helper method to call OpenAI with structured output"""
        try:
            system_prompt = self.prompt_manager.get_prompt(prompt_name)

            # Use instance temperature if not specified
            if temperature is None:
                temperature = self.temperature

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

    async def _analyze_section(
        self,
        ticker: str,
        section_name: str,
        query: str,
        prompt_name: str,
        response_model: Type[T],
        form_type: str = "10-K",
        limit: Optional[int] = None,
    ) -> T:
        """Generic method to analyze any document section"""
        # Use instance limit if not specified
        if limit is None:
            limit = self.document_limit

        # Retrieve documents
        documents = self.document_retriever.query_documents(
            query=query,
            ticker=ticker,
            form_type=form_type,
            limit=limit,
        )

        # Convert to context
        content = self.document_retriever.documents_to_context(documents)

        # Call OpenAI with structured output
        return await self._call_openai_structured(
            prompt_name=prompt_name,
            user_content=f"{section_name} content:\n{content}",
            response_model=response_model,
        )

    async def _analyze_section_from_config(
        self,
        ticker: str,
        analysis_type: str,
        section_key: str,
        response_model: Type[T],
        form_type: str = "10-K",
        limit: Optional[int] = None,
    ) -> T:
        """Analyze a section using configuration from YAML"""
        # Get config for this section
        config = self.config_loader.get_analysis_config(analysis_type, section_key)

        # Replace {ticker} placeholder in query if present
        query = config["query"].format(ticker=ticker)

        return await self._analyze_section(
            ticker=ticker,
            section_name=config["section_name"],
            query=query,
            prompt_name=config["prompt_name"],
            response_model=response_model,
            form_type=form_type,
            limit=limit,
        )
