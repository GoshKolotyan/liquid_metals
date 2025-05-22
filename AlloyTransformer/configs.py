import yaml
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Type, TypeVar, cast


@dataclass
class ModelConfig:
    # Required parameters (no defaults)
    train_path: str
    valid_path: str
    test_path: str
    seed: int
    learning_rate: float
    num_epochs: int
    batch_size: int
    num_workers: int
    patience: int
    gradient_clip: float
    model_version: str
    model_name: str
    description: str
    
    # Optional parameters (with defaults)
    feature_dim: Optional[int] = None
    d_model: int = 256
    num_head: int = 4
    num_transformer_layers: int = 3
    num_regression_head_layers: int = 2
    num_classification_head_layers: int = 1
    dropout: float = 0.1
    num_positions: int = 5
    dim_feedforward: int = 512
    use_property_focus: bool = True
    
    def as_dict(self) -> Dict[str, Any]: return asdict(self)


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