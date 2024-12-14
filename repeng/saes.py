import dataclasses
import json
import pathlib
import typing

import numpy as np
import torch
import torch.types
import tqdm


class SaeLayer(typing.Protocol):
    def encode(self, activation: np.ndarray) -> np.ndarray: ...
    def decode(self, features: np.ndarray) -> np.ndarray: ...


@dataclasses.dataclass
class Sae:
    layers: dict[int, SaeLayer]


def from_eleuther(
    repo_id: str,
    *,
    revision: str | None = None,
    device: str = "cpu",  # saes wants str | torch.device, safetensors wants str | int... so str it is
    dtype: torch.dtype | None = torch.bfloat16,
    layers: typing.Iterable[int] = range(1, 32),
) -> Sae:
    """
    Note that `layers` should be 1-indexed, repeng style, not 0-indexed, Eleuther style. This may change in the future.

    (Context: repeng counts embed_tokens as layer 0, then the first transformer block as layer 1, etc. Eleuther
    counts embed_tokens separately, then the first transformer block as layer 0.)
    """

    try:
        import huggingface_hub
        import safetensors.torch
        import sae as eleuther_sae  # type: ignore
    except ImportError as e:
        raise ImportError(
            "`sae` (or a transitive dependency) not installed"
            "--please install `sae` and its dependencies from https://github.com/EleutherAI/sae"
        ) from e

    @dataclasses.dataclass
    class EleutherSaeLayer:
        # see docstr
        # hang on to both for debugging
        repeng_layer: int
        eleuther_layer: int
        sae: eleuther_sae.Sae

        def encode(self, activation: np.ndarray) -> np.ndarray:
            # TODO: this materializes the entire, massive feature vector in memory
            # ideally, we would sparsify like the sae library does--need to figure out how to run PCA on the sparse matrix
            at = torch.from_numpy(activation).to(self.sae.device)
            out = self.sae.pre_acts(at)
            # numpy doesn't like bfloat16
            return out.cpu().float().numpy()

        def decode(self, features: np.ndarray) -> np.ndarray:
            # TODO: see encode, this is not great. `sae` ships with kernels for doing this sparsely, we should use them
            ft = torch.from_numpy(features).to(self.sae.device, dtype=dtype)
            decoded = ft @ self.sae.W_dec.mT.T
            return decoded.cpu().float().numpy()

    # TODO: only download requested layers?
    base_path = pathlib.Path(
        huggingface_hub.snapshot_download(repo_id, revision=revision)
    )
    layer_dict: dict[int, SaeLayer] = {}
    for layer in tqdm.tqdm(layers):
        eleuther_layer = layer - 1  # see docstr
        # this is in `sae` but to load the dtype we want, need to reimpl some stuff
        layer_path = base_path / f"layers.{eleuther_layer}"
        with (layer_path / "cfg.json").open() as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            try:
                # param removed in SAE lib but not in uploaded HF configs
                del cfg_dict["signed"]
            except KeyError as _:
                # for when they fix it eventually
                pass
            cfg = eleuther_sae.SaeConfig(**cfg_dict)

        layer_sae = eleuther_sae.Sae(d_in, cfg, device=device, dtype=dtype)
        safetensors.torch.load_model(
            model=layer_sae,
            filename=layer_path / "sae.safetensors",
            device=device,
            strict=True,
        )
        # repeng counts embed_tokens as layer 0 and further layers as 1, 2, ...
        # eleuther counts embed_tokens separately and further layers as 0, 1, ...
        layer_dict[layer] = EleutherSaeLayer(
            repeng_layer=layer, eleuther_layer=eleuther_layer, sae=layer_sae
        )

    return Sae(layers=layer_dict)
