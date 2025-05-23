import os
import sys
import hydra
import launch.prepare
from omegaconf import DictConfig
from hydra.utils import instantiate
import logging
from utils.util import set_seed

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path='configs', config_name='main')
def main(cfg: DictConfig):
    seed = cfg.get("seed", 42)
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    return _main(cfg)


def _main(cfg):
    logger.info(f"Original working directory: {hydra.utils.get_original_cwd()}")

    logger.info("Instantiating trainer")
    trainer = instantiate(cfg.trainer, task=cfg.task, _recursive_=False)
    logger.info("Done instantiating trainer")

    # set dataloader in trainer
    logger.info("Instantiating data module")
    data_module = instantiate(cfg.data, seed=cfg.get("seed", 42), _recursive_=False)
    trainer.set_data_module(data_module)

    if cfg.stage == "fit":
        logger.info("Model Fitting ...")
        trainer.fit()
        logger.info("Fitting done")
    else:  # test stage
        logger.info("Model Testing ...")
        trainer.test()
        logger.info("Testing done")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    main()