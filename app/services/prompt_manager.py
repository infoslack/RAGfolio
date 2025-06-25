from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class PromptManager:
    """Service responsible for loading and caching prompts"""

    def __init__(self, prompts_dir: Path = None):
        if prompts_dir is None:
            # Default to app/prompts directory
            prompts_dir = Path(__file__).parent.parent / "prompts"

        self.prompts_dir = prompts_dir
        self._prompt_cache: Dict[str, str] = {}

        # Ensure prompts directory exists
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def get_prompt(self, name: str) -> str:
        """Load prompt with lazy loading and caching"""
        if name not in self._prompt_cache:
            self._load_prompt(name)
        return self._prompt_cache[name]

    def _load_prompt(self, name: str) -> None:
        """Load a prompt file into cache"""
        prompt_file = self.prompts_dir / f"{name}.md"

        if not prompt_file.exists():
            error_msg = f"Prompt file not found: {name}.md in {self.prompts_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            self._prompt_cache[name] = prompt_file.read_text(encoding="utf-8")
            logger.debug(f"Loaded prompt: {name}.md")
        except Exception as e:
            error_msg = f"Failed to read prompt {name}.md: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def reload_prompt(self, name: str) -> str:
        """Force reload a prompt from disk"""
        if name in self._prompt_cache:
            del self._prompt_cache[name]
        return self.get_prompt(name)

    def clear_cache(self) -> None:
        """Clear all cached prompts"""
        self._prompt_cache.clear()
        logger.info("Cleared prompt cache")

    def list_available_prompts(self) -> List[str]:
        """List all available prompt files"""
        prompt_files = self.prompts_dir.glob("*.md")
        return [f.stem for f in prompt_files]
