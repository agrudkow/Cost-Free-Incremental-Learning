from typing import Optional
from pydantic import BaseSettings

from datasets import NAMES as DATASET_NAMES


class WAndBConfig(BaseSettings):
    project_name: str
    entity: str
    experiment_suffix: Optional[str]
    experiment_name: str
    group: str


class Config(BaseSettings):
    wandb: WAndBConfig
    '''Weight&Biases config'''
    tensorboard: bool
    '''Enable tensorboard logging'''
    dataset: DATASET_NAMES

