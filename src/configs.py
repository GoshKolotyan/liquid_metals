import yaml
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Type, TypeVar, cast


@dataclass
class ModelConfig:
    train_path: str
    valid_path: str
    test_path: str
    
    seed: int
    learning_rate: float
    num_epochs: int
    batch_size: int
    num_workers: int
    
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    max_seq_length: int
    
    patience: int
    gradient_clip: float
    
    model_version: str
    model_name: str
    description: str
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the ModelConfig object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the ModelConfig.
        """
        return asdict(self)


T = TypeVar('T')

class ConfigLoader:
    
    @staticmethod
    def load(config_path: str, config_class: Type[T] = ModelConfig) -> T:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if isinstance(config_dict, dict) and 'config' in config_dict:
            config_dict = config_dict['config']
        
        return cast(T, config_class(**config_dict))