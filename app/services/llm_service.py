from typing import List, AsyncGenerator
from groq import AsyncGroq  # Changed to async for consistency
from app.models.embeddings import Document
from app.config.settings import Settings
from app.services.prompt_manager import PromptManager
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, settings: Settings):
        self.client = AsyncGroq(api_key=settings.llm_api_key)  # Changed to AsyncGroq
        self.default_model = settings.llm_model
        self.default_temperature = settings.llm_temperature
        self.default_max_output_tokens = settings.llm_max_output_tokens

        # Initialize prompt manager to load system prompt
        prompts_dir = Path(__file__).parent.parent / "prompts"
        self.prompt_manager = PromptManager(prompts_dir)

    async def generate_response(  # Added async
        self,
        query: str,
        context_documents: List[Document],
        model: str = None,
        temperature: float = None,
        max_output_tokens: int = None,
    ) -> str:
        model = model or self.default_model
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        max_output_tokens = max_output_tokens or self.default_max_output_tokens

        context = "\n\n".join([doc.page_content for doc in context_documents])

        # Load system prompt from prompts directory
        system_prompt_template = self.prompt_manager.get_prompt("rag_response")
        system_prompt = system_prompt_template.format(context=context, query=query)

        try:
            completion = await self.client.chat.completions.create(  # Added await
                model=model,
                messages=[{"role": "system", "content": system_prompt}],
                temperature=temperature,
                max_tokens=max_output_tokens,
            )

            return completion.choices[0].message.content

        except Exception as e:
            logger.error(
                "LLM response generation failed",
                extra={"error": str(e), "query": query},
            )
            raise Exception(f"Failed to generate response: {str(e)}")

    # Stream API
    async def generate_stream_response(
        self,
        query: str,
        context_documents: List[Document],
        model: str = None,
        temperature: float = None,
        max_output_tokens: int = None,
    ) -> AsyncGenerator[str, None]:
        model = model or self.default_model
        temperature = (
            temperature if temperature is not None else self.default_temperature
        )
        max_output_tokens = max_output_tokens or self.default_max_output_tokens

        context = "\n\n".join([doc.page_content for doc in context_documents])

        # Load system prompt from prompts directory
        system_prompt_template = self.prompt_manager.get_prompt("rag_response")
        system_prompt = system_prompt_template.format(context=context, query=query)

        try:
            # Create streaming response
            stream = await self.client.chat.completions.create(  # Added await
                model=model,
                messages=[{"role": "system", "content": system_prompt}],
                temperature=temperature,
                max_tokens=max_output_tokens,
                stream=True,  # Enable streaming
            )

            # Process stream events
            async for chunk in stream:  # Changed to async for
                if chunk.choices[0].delta.content is not None:
                    yield json.dumps(
                        {
                            "type": "text_delta",
                            "delta": chunk.choices[0].delta.content,
                        }
                    )

            # Send completion event
            yield json.dumps({"type": "stream_completed"})

        except Exception as e:
            logger.error(
                "LLM stream response generation failed",
                extra={"error": str(e), "query": query},
            )
            yield json.dumps(
                {
                    "type": "error",
                    "message": f"Failed to generate stream response: {str(e)}",
                }
            )
