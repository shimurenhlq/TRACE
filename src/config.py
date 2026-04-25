"""
Configuration management for TRACE system.
Handles model endpoints, API keys, and retrieval parameters.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """Configuration for a single model endpoint."""
    provider: str  # "openai", "aliyun", "anthropic", "local"
    model: str
    api_key: str
    base_url: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None


class Config:
    """Main configuration class for TRACE system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file or environment variables.

        Args:
            config_path: Path to config.yaml file. If None, uses environment variables.
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            self._load_from_dict(config_dict)
        else:
            self._load_from_env()

    def _load_from_dict(self, config_dict: Dict):
        """Load configuration from dictionary."""
        # Model configurations
        models = config_dict.get('models', {})

        self.planner = self._parse_model_config(
            models.get('planner', {}),
            default_provider='openai',
            default_model='gpt-4'
        )

        self.navigator = self._parse_model_config(
            models.get('navigator', {}),
            default_provider='openai',
            default_model='gpt-4-vision-preview'
        )

        self.reasoner = self._parse_model_config(
            models.get('reasoner', {}),
            default_provider='openai',
            default_model='gpt-4-vision-preview'
        )

        # ColPali configuration
        colpali_config = config_dict.get('colpali', {})
        self.colpali_model_path = colpali_config.get('model_path', 'vidore/colpali-v1.2')
        self.colpali_device = colpali_config.get('device', 'cuda')

        # Retrieval parameters
        retrieval = config_dict.get('retrieval', {})
        self.top_k = retrieval.get('top_k', 10)
        self.graph_threshold = retrieval.get('graph_threshold', 0.7)
        self.max_pages_per_step = retrieval.get('max_pages_per_step', 3)
        self.graph_k_neighbors = retrieval.get('graph_k_neighbors', 3)

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Planner
        self.planner = ModelConfig(
            provider=os.getenv('PLANNER_PROVIDER', 'openai'),
            model=os.getenv('PLANNER_MODEL', 'gpt-4'),
            api_key=os.getenv('PLANNER_API_KEY', os.getenv('OPENAI_API_KEY', '')),
            base_url=os.getenv('PLANNER_BASE_URL'),
            model_info={'vision': False, 'function_calling': True, 'json_output': True}
        )

        # Navigator
        self.navigator = ModelConfig(
            provider=os.getenv('NAVIGATOR_PROVIDER', 'openai'),
            model=os.getenv('NAVIGATOR_MODEL', 'gpt-4-vision-preview'),
            api_key=os.getenv('NAVIGATOR_API_KEY', os.getenv('OPENAI_API_KEY', '')),
            base_url=os.getenv('NAVIGATOR_BASE_URL'),
            model_info={'vision': True, 'function_calling': False, 'json_output': False}
        )

        # Reasoner
        self.reasoner = ModelConfig(
            provider=os.getenv('REASONER_PROVIDER', 'openai'),
            model=os.getenv('REASONER_MODEL', 'gpt-4-vision-preview'),
            api_key=os.getenv('REASONER_API_KEY', os.getenv('OPENAI_API_KEY', '')),
            base_url=os.getenv('REASONER_BASE_URL'),
            model_info={'vision': True, 'function_calling': False, 'json_output': False}
        )

        # ColPali
        self.colpali_model_path = os.getenv('COLPALI_MODEL_PATH', 'vidore/colpali-v1.2')
        self.colpali_device = os.getenv('COLPALI_DEVICE', 'cuda')

        # Retrieval parameters
        self.top_k = int(os.getenv('RETRIEVAL_TOP_K', '10'))
        self.graph_threshold = float(os.getenv('GRAPH_THRESHOLD', '0.7'))
        self.max_pages_per_step = int(os.getenv('MAX_PAGES_PER_STEP', '3'))
        self.graph_k_neighbors = int(os.getenv('GRAPH_K_NEIGHBORS', '3'))

    def _parse_model_config(self, model_dict: Dict, default_provider: str, default_model: str) -> ModelConfig:
        """Parse model configuration from dictionary."""
        provider = model_dict.get('provider', default_provider)
        model = model_dict.get('model', default_model)
        api_key = model_dict.get('api_key', '')

        # Handle environment variable substitution
        if api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var, '')

        base_url = model_dict.get('base_url')
        if base_url and base_url.startswith('${') and base_url.endswith('}'):
            env_var = base_url[2:-1]
            base_url = os.getenv(env_var)

        model_info = model_dict.get('model_info', {})

        return ModelConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            model_info=model_info
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'models': {
                'planner': {
                    'provider': self.planner.provider,
                    'model': self.planner.model,
                    'base_url': self.planner.base_url
                },
                'navigator': {
                    'provider': self.navigator.provider,
                    'model': self.navigator.model,
                    'base_url': self.navigator.base_url
                },
                'reasoner': {
                    'provider': self.reasoner.provider,
                    'model': self.reasoner.model,
                    'base_url': self.reasoner.base_url
                }
            },
            'colpali': {
                'model_path': self.colpali_model_path,
                'device': self.colpali_device
            },
            'retrieval': {
                'top_k': self.top_k,
                'graph_threshold': self.graph_threshold,
                'max_pages_per_step': self.max_pages_per_step,
                'graph_k_neighbors': self.graph_k_neighbors
            }
        }
