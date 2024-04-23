from pathlib import Path

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, SequentialLR
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, LlamaTokenizer

from datasets.gqa import GQA, GQAEval, GQACollate
from datasets.mmvp import MMVP, MMVPCollate, MMVPEval
from datasets.seedbench import SEEDBenchSingleImage, SEEDBenchSingleImageEval, SEEDBenchCollate
from datasets.whatsup import WhatsUp, WhatsUpEval, WhatsUpCollate
from models.cogvlm import *
from models.wrappers import *


def create_model(cfg, device):
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
                                                                  attn_implementation=cfg.model.attn_impl,
                                                                  low_cpu_mem_usage=True,
                                                                  )
            processor = AutoProcessor.from_pretrained(cfg.model.processor_path)
            model = GenerativeWrapper(processor, llava, device, dtype)

        case "cogvlm":
            cogvlm = AutoModelForCausalLM.from_pretrained(
                cfg.model.path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            tokenizer = LlamaTokenizer.from_pretrained(cfg.model.tokenizer_path)
            processor = CogVLMProcessor(tokenizer, cogvlm.config)
            # Horrible monkey-patching because the original implementation does not support image batching as tensor
            cogvlm.model.encode_images = encode_images_replacement.__get__(cogvlm.model, cogvlm.model.__class__)
            model = GenerativeWrapper(processor, cogvlm, device, dtype)
        case _:
            raise ValueError(f"Unsupported model '{cfg.model.name}'")

    return model.to(device)


def create_eval_task(cfg, device):
    sampling_config = GenerationConfig(**cfg.sampling_params)
    match cfg.task.name:
        case "SEED-Bench 2":
            dataset = SEEDBenchSingleImage(cfg.task.task_num, Path(cfg.task.json_path), Path(cfg.task.image_root))
            wrapper = SEEDBenchSingleImageEval(cfg.prompt.text, device, cfg.task.eval_method)
            collate = SEEDBenchCollate()

        case "WhatsUp":
            if cfg.task.part == 'A':
                json_path = Path(cfg.task.json_path_A)
            elif cfg.task.part == 'B':
                json_path = Path(cfg.task.json_path_B)
            else:
                raise ValueError(f"Unsupported What's Up part '{cfg.task.part}'")

            dataset = WhatsUp(Path(cfg.task.image_root), json_path, permute_options=cfg.task.permute)
            wrapper = WhatsUpEval(cfg.prompt.text, device, cfg.task.eval_method)
            collate = WhatsUpCollate()

        case "GQA":
            dataset = GQA(Path(cfg.task.test_question_file), Path(cfg.task.img_dir))
            wrapper = GQAEval(cfg.prompt.text, device, cfg.task.eval_method, sampling_config)
            collate = GQACollate()

        case "MMVP":
            dataset = MMVP(Path(cfg.task.csv_path), Path(cfg.task.img_dir))
            wrapper = MMVPEval(cfg.prompt.text, cfg.task.eval_method)
            collate = MMVPCollate()

        case _:
            raise ValueError(f"Unsupported task '{cfg.task.name}'")

    return dataset, wrapper, collate


def create_optimizer(cfg, model):
    match cfg.optim:
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        case "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        case "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        case _:
            raise ValueError(f"Optimizer {cfg.optim} not supported")

    schedulers = []
    if cfg.warmup_epochs > 0:
        schedulers.append(
            LinearLR(optimizer, start_factor=0.001, end_factor=1, total_iters=cfg.warmup_epochs, verbose=True))
    if cfg.annealing_t0 > 0:
        schedulers.append(CosineAnnealingWarmRestarts(optimizer, T_0=cfg.annealing_t0, verbose=True))
    if cfg.scheduler_patience > 0:
        schedulers.append(ReduceLROnPlateau(optimizer, patience=cfg.scheduler_patience, verbose=True))
    match len(schedulers):
        case 0:
            scheduler = None
        case 1:
            scheduler = schedulers[0]
        case _:
            scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=[cfg.warmup_epochs])
    return optimizer, scheduler


def create_train_task(cfg):
    match cfg.task.name:
        case "GQA":
            train_dataset = GQA(Path(cfg.task.train_question_file), Path(cfg.task.img_dir), cfg.task.train_size)
            val_dataset = GQA(Path(cfg.task.val_question_file), Path(cfg.task.img_dir), cfg.task.val_size)
            test_dataset = GQA(Path(cfg.task.test_question_file), Path(cfg.task.img_dir), cfg.task.test_size)
            collate_fn = None

        case _:
            raise ValueError(f"Unsupported task '{cfg.task.name}'")

    return train_dataset, val_dataset, test_dataset, train_wrapper, val_wrapper, test_wrapper, collate_fn
