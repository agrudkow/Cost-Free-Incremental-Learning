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
    synth_img_optim_steps: int
    synth_img_optim_lr: float
    synth_img_save_dir: Optional[str]
    synth_img_save_num: Optional[int]
    dirichlet_max_iter: Optional[int]


class Config(BaseSettings):
    wandb: WAndBConfig
    cf_il: CFILConfig
