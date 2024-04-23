import torch
from PIL.Image import Image
from torchvision import transforms
from transformers.feature_extraction_utils import BatchFeature


def encode_images_replacement(self, images: torch.Tensor) -> torch.Tensor:
    images_features = self.vision(images)
    return images_features


class CogVLMProcessor:
    """
    All credit to the original CogVLM huggingface script
    """
    LANGUAGE_TOKEN_TYPE = 0
    VISION_TOKEN_TYPE = 1

    def __init__(self, tokenizer, model_config):
        self.tokenizer = tokenizer
        self.config = model_config
        self.image_size: int = self.config.vision_config['image_size']
        self.patch_size: int = self.config.vision_config['patch_size']
        self.img_to_tensors = transforms.ToTensor()
        self.img_transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def __call__(self, text: list[str], images: list[Image], padding=True, **_):
        text_inputs = self.tokenizer(text, return_tensors=None, padding=False, truncation=False,
                                     add_special_tokens=False)
        batch_size = len(text)

        transformed_images = torch.stack([self.img_to_tensors(img) for img in images])
        transformed_images = self.img_transform(transformed_images)
        # Patches without CLS, with BOI and EOI tokens
        vision_token_num = (self.image_size // self.patch_size) * (self.image_size // self.patch_size) + 2

        input_ids = [
            [self.tokenizer.bos_token_id]
            + [self.tokenizer.pad_token_id] * vision_token_num
            + text_inputs.input_ids[i]
            for i in range(batch_size)
        ]
        token_type_ids = [
            [self.LANGUAGE_TOKEN_TYPE]
            + [self.VISION_TOKEN_TYPE] * vision_token_num
            + [self.LANGUAGE_TOKEN_TYPE] * len(text_inputs.input_ids[i])
            for i in range(batch_size)
        ]

        attention_mask = [[1] * len(input_ids[i]) for i in range(batch_size)]

        if padding:
            max_length = max(len(x) for x in input_ids)
            input_ids = [[self.tokenizer.pad_token_id] * (max_length - len(x)) + x for x in input_ids]
            token_type_ids = [[self.LANGUAGE_TOKEN_TYPE] * (max_length - len(x)) + x for x in token_type_ids]
            attention_mask = [[0] * (max_length - len(x)) + x for x in attention_mask]

        return BatchFeature(data={
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'images': transformed_images,
        })
