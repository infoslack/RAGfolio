from typing import Optional
from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)


class TickerExtractor:
    """Service responsible for extracting ticker symbols from user messages"""

    def __init__(self, openai_api_key: str, model: str, prompt_manager):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.prompt_manager = prompt_manager

        # Company to ticker mapping for fast lookup
        self.company_map = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "meta": "META",
            "facebook": "META",
            "netflix": "NFLX",
            "nvidia": "NVDA",
            "disney": "DIS",
            "coca cola": "KO",
            "ibm": "IBM",
            "johnson": "JNJ",
        }

    async def extract_ticker(self, message: str) -> Optional[str]:
        """Extract ticker symbol from user message using mapping + LLM fallback"""

        # First try: Direct mapping (fast)
        ticker = self._try_direct_mapping(message)
        if ticker:
            return ticker

        # Second try: LLM extraction (slower but comprehensive)
        return await self._try_llm_extraction(message)

    def _try_direct_mapping(self, message: str) -> Optional[str]:
        """Try to find ticker using direct company name mapping"""
        message_lower = message.lower()

        for company, ticker in self.company_map.items():
            if company in message_lower:
                logger.info(f"Direct mapping found ticker: {ticker}")
                return ticker

        return None

    async def _try_llm_extraction(self, message: str) -> Optional[str]:
        """Use LLM to extract ticker from message"""
        try:
            logger.info("Using LLM to extract ticker from message")

            extraction_prompt = self.prompt_manager.get_prompt("ticker_extraction")

            completion = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": f"Extract ticker from: {message}"},
                ],
                max_tokens=10,
            )

            result = completion.choices[0].message.content.strip().upper()

            if result == "NONE" or len(result) > 6:  # Basic validation
                logger.info("LLM could not extract valid ticker")
                return None

            logger.info(f"LLM extracted ticker: {result}")
            return result

        except Exception as e:
            logger.error(f"LLM ticker extraction failed: {str(e)}")
            return None
