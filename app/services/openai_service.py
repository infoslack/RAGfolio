from typing import List, AsyncGenerator
from openai import OpenAI
from app.models.embeddings import Document
from app.config.settings import Settings
import logging
import json

logger = logging.getLogger(__name__)


class OpenAIService:
    def __init__(self, settings: Settings):
        self.base_client = OpenAI(api_key=settings.openai_api_key)
        self.default_model = settings.openai_model
        self.default_temperature = settings.openai_temperature
        self.default_max_output_tokens = settings.openai_max_output_tokens
        self.system_prompt_template = settings.openai_system_prompt

    def generate_response(
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

        prompt = prompt = self.system_prompt_template.format(
            context=context, query=query
        )

        try:
            response = self.client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            return response.output_text

        except Exception as e:
            logger.error(
                "OpenAI response generation failed",
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

        prompt = self.system_prompt_template.format(context=context, query=query)

        try:
            # Create streaming response
            stream = self.client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stream=True,  # Enable streaming
            )

            # Process stream events
            for event in stream:
                if hasattr(event, "type"):
                    event_type = event.type

                    # Handle different event types based on the documentation
                    if event_type == "response.created":
                        yield json.dumps(
                            {
                                "type": "response.created",
                                "response_id": event.response.id,
                                "model": event.response.model,
                            }
                        )

                    elif event_type == "response.output_text.delta":
                        # This is the main event for text streaming
                        yield json.dumps(
                            {
                                "type": "text_delta",
                                "delta": event.delta,
                                "output_index": event.output_index,
                                "content_index": event.content_index,
                            }
                        )

                    elif event_type == "response.output_text.done":
                        yield json.dumps(
                            {
                                "type": "text_done",
                                "text": event.text,
                                "output_index": event.output_index,
                                "content_index": event.content_index,
                            }
                        )

                    elif event_type == "response.completed":
                        yield json.dumps(
                            {
                                "type": "response.completed",
                                "response_id": event.response.id,
                                "usage": event.response.usage.model_dump()
                                if event.response.usage
                                else None,
                            }
                        )

                    elif event_type == "response.failed":
                        error_msg = (
                            event.response.error.message
                            if event.response.error
                            else "Unknown error"
                        )
                        yield json.dumps(
                            {"type": "response.failed", "error": error_msg}
                        )
                        logger.error(
                            "OpenAI stream failed",
                            extra={"error": error_msg, "query": query},
                        )
                        break

        except Exception as e:
            logger.error(
                "OpenAI stream response generation failed",
                extra={"error": str(e), "query": query},
            )
            yield json.dumps(
                {
                    "type": "error",
                    "message": f"Failed to generate stream response: {str(e)}",
                }
            )
