import torch
import wandb
from tqdm.auto import tqdm

from datasets.base import BaseDataset


@torch.inference_mode()
def eval_model(model, eval_wrapper, dataloader, show_tqdm=False):
    model.eval()
    size = 0
    total_matches = 0
    results = []
    for batch in tqdm(dataloader, disable=not show_tqdm):
        res: EvaluationResult = eval_wrapper(batch, model)
        results.append(res)
        total_matches += res.num_matches()
        size += res.batch_size
    return total_matches / size, results


from dataclasses import dataclass

import torch


@dataclass
class EvaluationResult:
    batch_size: int
    question_ids: torch.Tensor
    predictions: torch.Tensor | list[str] | list[tuple]
    labels: torch.Tensor | list[str] | list[tuple]

    def matches(self):
        matches = []
        if isinstance(self.predictions, torch.Tensor):
            matches = (self.predictions == self.labels)
        elif isinstance(self.predictions[0], str):
            for pred, label in zip(self.predictions, self.labels):
                if pred.strip().lower() == label.strip().lower():
                    matches.append(1)
                else:
                    matches.append(0)
        else:
            for pred, label in zip(self.predictions, self.labels):
                if pred == label:
                    matches.append(1)
                else:
                    matches.append(0)
        return matches

    def num_matches(self):
        return sum(self.matches())


def merge_results(results):
    question_ids = []
    predictions = []
    labels = []
    for res in results:
        question_ids.extend(res.question_ids)
        predictions.extend(res.predictions)
        labels.extend(res.labels)

    if isinstance(predictions[0], torch.Tensor):
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
    return EvaluationResult(
        batch_size=sum(res.batch_size for res in results),
        question_ids=torch.tensor(question_ids),
        predictions=predictions,
        labels=labels,
    )


def sample_results(results: EvaluationResult, dataset: BaseDataset, num_samples: int):
    matches = torch.BoolTensor(results.matches())
    correct_ind = results.question_ids[matches][:num_samples].tolist()
    mistakes_ind = results.question_ids[~matches][:num_samples].tolist()
    correct = [dataset.wandb_repr(i, results.predictions[i]) for i in correct_ind]
    mistakes = [dataset.wandb_repr(i, results.predictions[i]) for i in mistakes_ind]
    headers = dataset.wandb_columns
    return wandb.Table(data=correct, columns=headers), wandb.Table(data=mistakes, columns=headers)
