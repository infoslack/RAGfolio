from typing import Optional
from openai import AsyncOpenAI
from app.utils.decorators import handle_errors
import logging

logger = logging.getLogger(__name__)


class TickerExtractor:
    """Service responsible for extracting ticker symbols from user messages"""

    def __init__(
        self,
        openai_api_key: str,
        model: str,
        prompt_manager,
        config_loader,
        temperature: float = 0.0,
        max_tokens: int = 10,
    ):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.prompt_manager = prompt_manager
        self.config_loader = config_loader
        self.temperature = temperature
        self.max_tokens = max_tokens

    @handle_errors("Ticker extraction")
    async def extract_ticker(self, message: str) -> Optional[str]:
        """Extract ticker symbol from user message using mapping + LLM fallback"""

        # First try: Direct mapping (fast)
        ticker = self._try_direct_mapping(message)
        if ticker:
            return ticker

        # Second try: LLM extraction (slower but comprehensive)
        return await self._try_llm_extraction(message)

    def _try_direct_mapping(self, message: str) -> Optional[str]:
        """Try to find ticker using direct company name mapping from config"""
        message_lower = message.lower()

        # Get mappings from config
        company_map = self.config_loader.get_ticker_mappings()

        for company, ticker in company_map.items():
            if company.lower() in message_lower:
                logger.info(f"Direct mapping found ticker: {ticker}")
                return ticker

        return None

    @handle_errors("LLM ticker extraction")
    async def _try_llm_extraction(self, message: str) -> Optional[str]:
        """Use LLM to extract ticker from message"""
        logger.info("Using LLM to extract ticker from message")

        extraction_prompt = self.prompt_manager.get_prompt("ticker_extraction")

        completion = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": f"Extract ticker from: {message}"},
            ],
            max_tokens=self.max_tokens,
        )

        result = completion.choices[0].message.content.strip().upper()

        if result == "NONE" or len(result) > 6:  # Basic validation
            logger.info("LLM could not extract valid ticker")
            return None

        logger.info(f"LLM extracted ticker: {result}")
        return result
