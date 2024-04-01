import os
from pathlib import Path

import hydra
import torch
import wandb
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from utils.eval import eval_model, merge_results, sample_results
from utils.misc import set_random_seed, count_parameters
from utils.setup import create_model, create_train_task


@hydra.main(config_path="configs", config_name="train_config", version_base=None)
def run(cfg: DictConfig):
    torch.autograd.set_detect_anomaly(cfg.detect_anomalies, check_nan=True)
    torch.backends.cuda.matmul.allow_tf32 = cfg.use_tf32
    torch.backends.cudnn.allow_tf32 = cfg.use_tf32

    set_random_seed(cfg.seed)

    accelerator = Accelerator()

    if cfg.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        print(OmegaConf.to_yaml(cfg))

    if cfg.resume_wandb_id is None:
        run_id = wandb.util.generate_id()
    else:
        run_id = cfg.resume_wandb_id

    print(f"Run id: {run_id}")

    choices = HydraConfig.get().runtime.choices
    root_dir = Path(__file__).parent.resolve()
    prompt_path = root_dir.joinpath(root_dir, "configs", "prompts",
                                    f"{choices['model']}_{choices['task']}_{cfg.task.eval_method}.yaml")
    cfg.prompt = OmegaConf.load(prompt_path)

    cfg.device = accelerator.device

    dataset, wrapper, collate_fn = create_train_task(cfg)
    model = create_model(cfg)

    cfg.model_size = count_parameters(model)
    wandb.init(id=run_id, resume="must" if cfg.resume_wandb_id is not None else "never",
               project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.name, group=cfg.task.display_name,
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    wandb.watch(model)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=cfg.pin_memory, collate_fn=collate_fn)

    accuracy, results = eval_model(model, wrapper, loader, cfg.show_tqdm)
    merged_results = merge_results(results)
    correct_table, mistakes_table = sample_results(merged_results, dataset, cfg.log_samples)
    if cfg.disable_wandb:
        print({"accuracy": accuracy, "correct": correct_table, "mistakes": mistakes_table})
    else:
        wandb.log({"correct": correct_table, "mistakes": mistakes_table})
        wandb.run.summary.update({"accuracy": accuracy})


if __name__ == '__main__':
    run()
