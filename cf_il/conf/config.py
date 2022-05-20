from typing import Optional, Tuple
from pydantic import BaseSettings


class WAndBConfig(BaseSettings):
    project_name: str
    entity: str
    experiment_suffix: Optional[str]
    experiment_name: str
    group: str


class CFILConfig(BaseSettings):
    dataset: str
    tensorboard: bool
    validation: bool
    n_epochs: int
    lr: float
    momentum: float
    batch_size: int
    buffer_size: int
    buffer_batch_size: int
    alpha: float
    eta: float
    tau: float
    scale: Tuple[float, float]


class Config(BaseSettings):
    wandb: WAndBConfig
    cf_il: CFILConfig
