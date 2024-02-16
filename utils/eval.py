import torch
import wandb
from torch.utils.data import Dataset
from tqdm.auto import tqdm


@torch.inference_mode()
def eval_model(model, eval_wrapper, dataloader, show_tqdm=False):
    model.eval()
    size = 0
    total_matches = 0
    results = []
    for batch in tqdm(dataloader, disable=not show_tqdm):
        res: EvaluationResult = eval_wrapper(batch, model)
        results.append(res)
        total_matches += res.num_matches().item()
        size += res.batch_size
    return total_matches / size, results


from dataclasses import dataclass

import torch


@dataclass
class EvaluationResult:
    batch_size: int
    question_ids: torch.Tensor
    predictions: torch.Tensor
    labels: torch.Tensor

    def num_matches(self):
        return (self.predictions == self.labels).sum()


def merge_results(results):
    question_ids = []
    predictions = []
    labels = []
    for res in results:
        question_ids.append(res.question_ids)
        predictions.append(res.predictions)
        labels.append(res.labels)

    return EvaluationResult(
        batch_size=sum(res.batch_size for res in results),
        question_ids=torch.cat(question_ids),
        predictions=torch.cat(predictions),
        labels=torch.cat(labels),
    )


def sample_results(results: EvaluationResult, dataset, num_samples: int):
    matches = results.predictions == results.labels
    correct_ind = results.question_ids[matches][:num_samples].tolist()
    mistakes_ind = results.question_ids[~matches][:num_samples].tolist()
    correct = [dataset.wandb_repr(i, results.predictions[i].item()) for i in correct_ind]
    mistakes = [dataset.wandb_repr(i, results.predictions[i].item()) for i in mistakes_ind]
    headers = dataset.wandb_columns
    return wandb.Table(data=correct, columns=headers), wandb.Table(data=mistakes, columns=headers)
