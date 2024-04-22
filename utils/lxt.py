import lxt.functional as lf
import numpy as np


class InputsHook:
    def __init__(self, arg_ind=None, kwarg_name=None):

        if arg_ind is not None and kwarg_name is not None:
            raise ValueError("Only one of arg_ind and kwarg_name should be provided")

        if arg_ind is None and kwarg_name is None:
            raise ValueError("Either arg_ind or kwarg_name should be provided")

        self.inputs = None
        self.arg_ind = arg_ind
        self.kwarg_name = kwarg_name

    def __call__(self, _, args, kwargs):
        if self.arg_ind is not None:
            self.inputs = args[self.arg_ind]
        if self.kwarg_name is not None:
            self.inputs = kwargs[self.kwarg_name]


def calculate_relevance(logits, inputs, class_ind, img_begin, img_end, img_dims):
    probs = lf.softmax(logits, dim=-1)
    inputs.retain_grad()
    score = probs[class_ind]
    score.backward(score)
    relevance = inputs.grad.float().sum(-1).cpu().squeeze().numpy()
    # relevance = relevance / abs(relevance).max()
    image_relevance = relevance[img_begin:img_end].reshape(img_dims)
    text_relevance = np.concatenate((relevance[0:img_begin], relevance[img_end:]))
    inputs.grad = None
    return image_relevance, text_relevance, score
