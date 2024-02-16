from pathlib import Path

from transformers import LlavaForConditionalGeneration, AutoProcessor

from datasets.seedbench import SEEDBenchSingleImage, SEEDBenchSingleImageEval, SEEDBenchCollate
from datasets.whatsup import WhatsUp, WhatsUpEval, WhatsUpCollate
from models.wrappers import *


def create_model(cfg):
    match cfg.model.dtype:
        case "bfloat16":
            dtype = torch.bfloat16
        case "float16":
            dtype = torch.float16
        case "float32":
            dtype = torch.float32
        case _:
            raise ValueError(f"Unsupported dtype '{cfg.model.dtype}'")

    match cfg.model.name:
        case "llava":
            llava = LlavaForConditionalGeneration.from_pretrained(cfg.model.path, torch_dtype=dtype,
                                                                  attn_implementation=cfg.model.attn_impl
                                                                  ).to(cfg.device)
            processor = AutoProcessor.from_pretrained(cfg.model.processor_path)
            model = GenerativeWrapper(processor, llava, dtype)
        case _:
            raise ValueError(f"Unsupported model '{cfg.model.name}'")

    return model


def create_task(cfg):
    match cfg.task.name:
        case "SEED-Bench 2":
            dataset = SEEDBenchSingleImage(cfg.task.task_num, Path(cfg.task.json_path), Path(cfg.task.image_root))
            wrapper = SEEDBenchSingleImageEval(cfg.prompt.text, cfg.device, cfg.task.eval_method)
            collate = SEEDBenchCollate()

        case "WhatsUp":
            if cfg.task.part == 'A':
                json_path = Path(cfg.task.json_path_A)
            elif cfg.task.part == 'B':
                json_path = Path(cfg.task.json_path_B)
            else:
                raise ValueError(f"Unsupported What'sUp part '{cfg.task.part}'")

            dataset = WhatsUp(Path(cfg.task.image_root), json_path, permute_options=cfg.task.permute)
            wrapper = WhatsUpEval(cfg.prompt.text, cfg.device, cfg.task.eval_method)
            collate = WhatsUpCollate()
        case _:
            raise ValueError(f"Unsupported task '{cfg.task.name}'")

    return dataset, wrapper, collate
