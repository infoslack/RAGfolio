import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Service responsible for loading and caching configuration files"""

    def __init__(self, queries_path: str, ticker_mappings_path: str):
        self.queries_path = Path(queries_path)
        self.ticker_mappings_path = Path(ticker_mappings_path)

        # Cache for loaded configs
        self._queries_cache: Optional[Dict[str, Any]] = None
        self._ticker_mappings_cache: Optional[Dict[str, str]] = None

    def get_queries(self) -> Dict[str, Any]:
        """Load and cache queries configuration"""
        if self._queries_cache is None:
            self._queries_cache = self._load_yaml(self.queries_path)
            logger.info(f"Loaded queries configuration from {self.queries_path}")
        return self._queries_cache

    def get_ticker_mappings(self) -> Dict[str, str]:
        """Load and cache ticker mappings"""
        if self._ticker_mappings_cache is None:
            config = self._load_yaml(self.ticker_mappings_path)
            self._ticker_mappings_cache = config.get("company_ticker_mappings", {})
            logger.info(f"Loaded ticker mappings from {self.ticker_mappings_path}")
        return self._ticker_mappings_cache

    def get_analysis_config(self, analysis_type: str, section: str) -> Dict[str, str]:
        """Get configuration for a specific analysis section"""
        queries = self.get_queries()
        analysis_queries = queries.get("analysis_queries", {})

        if analysis_type not in analysis_queries:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        if section not in analysis_queries[analysis_type]:
            raise ValueError(
                f"Unknown section '{section}' for analysis type '{analysis_type}'"
            )

        return analysis_queries[analysis_type][section]

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file"""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load YAML file {path}: {e}")
            raise

    def reload_configs(self) -> None:
        """Force reload all configurations"""
        self._queries_cache = None
        self._ticker_mappings_cache = None
        logger.info("Cleared configuration cache")
