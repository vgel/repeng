import dataclasses

import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm


@dataclasses.dataclass
class DatasetEntry:
    positive: str
    negative: str


@dataclasses.dataclass
class ControlVector:
    directions: dict[int, np.ndarray]

    @classmethod
    def train(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[DatasetEntry],
        **kwargs,
    ) -> "ControlVector":
        dirs = read_representations(
            model,
            tokenizer,
            dataset,
            **kwargs,
        )
        return cls(dirs)


def read_representations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[DatasetEntry],
    hidden_layers: list[int] = [],
    batch_size: int = 32,
) -> dict[int, np.ndarray]:
    """
    Extract the representations based on the contrast dataset.
    """

    if not hidden_layers:
        hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))

    # the order is [positive, negative, positive, negative, ...]
    train_strs = [s for ex in inputs for s in (ex.positive, ex.negative)]

    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layers, batch_size
    )

    # get differences between (positive, negative) pairs
    relative_layer_hiddens = {}
    for layer in hidden_layers:
        relative_layer_hiddens[layer] = (
            layer_hiddens[layer][::2] - layer_hiddens[layer][1::2]
        )

    # get directions for each layer using PCA
    directions: dict[int, np.ndarray] = {}
    for layer in tqdm.tqdm(hidden_layers):
        assert layer_hiddens[layer].shape[0] == len(inputs) * 2

        # fit layer directions
        train = np.vstack(
            relative_layer_hiddens[layer]
            - relative_layer_hiddens[layer].mean(axis=0, keepdims=True)
        )
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        # shape (n_features,)
        directions[layer] = pca_model.components_.astype(np.float32).squeeze(axis=0)

        # calculate sign
        projected_hiddens = project_onto_direction(
            layer_hiddens[layer], directions[layer]
        )

        # order is [positive, negative, positive, negative, ...]
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )

        if positive_smaller_mean > positive_larger_mean:  # type: ignore
            directions[layer] *= -1

    return directions


def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    """
    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the last token.

    Returns a dictionary from `hidden_layers` layer id to an numpy array of shape `(n_inputs, hidden_dim)`
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layers}
    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs):
            out = model(
                **tokenizer(batch, padding=True, return_tensors="pt").to(model.device),
                output_hidden_states=True,
            )
            for layer in hidden_layers:
                for batch in out.hidden_states[layer]:
                    hidden_states[layer].append(batch[-1, :].squeeze().cpu().numpy())
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag
