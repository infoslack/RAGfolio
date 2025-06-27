from typing import Optional
from groq import AsyncGroq
import instructor
from pydantic import BaseModel
from app.utils.decorators import handle_errors
import logging

logger = logging.getLogger(__name__)


class TickerResponse(BaseModel):
    """Pydantic model for ticker extraction response"""

    ticker: Optional[str]
    reasoning: str


class TickerExtractor:
    """Service responsible for extracting ticker symbols from user messages"""

    def __init__(
        self,
        llm_api_key: str,
        model: str,
        prompt_manager,
        config_loader,
        temperature: float = 0.0,
        max_tokens: int = 50,  # Increased for reasoning
    ):
        # Initialize client and patch with Instructor
        base_client = AsyncGroq(api_key=llm_api_key)
        self.client = instructor.from_groq(base_client)
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

        # Second try: LLM extraction with Instructor (slower but comprehensive)
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
        """Use LLM with Instructor to extract ticker from message"""
        logger.info("Using LLM to extract ticker from message")

        extraction_prompt = self.prompt_manager.get_prompt("ticker_extraction")

        # Use Instructor for structured extraction
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": f"Extract ticker from: {message}"},
            ],
            response_model=TickerResponse,  # Instructor handles validation
        )

        # Validate ticker
        if not response.ticker or response.ticker == "NONE" or len(response.ticker) > 6:
            logger.info(
                f"LLM could not extract valid ticker. Reasoning: {response.reasoning}"
            )
            return None

        logger.info(
            f"LLM extracted ticker: {response.ticker}. Reasoning: {response.reasoning}"
        )
        return response.ticker.upper()
