import dataclasses
import typing
import warnings

import torch
from transformers import PretrainedConfig, PreTrainedModel

from repeng.utils import get_num_hidden_layer

if typing.TYPE_CHECKING:
    from .extract import ControlVector


class ControlModel(torch.nn.Module):
    """
    **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

    A wrapped language model that can have controls set on its layers with `self.set_control`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        layer_ids: typing.Optional[typing.Iterable[int]]=None,
        layer_zones: typing.Optional[typing.Iterable[float]]=None,
    ):
        """
        **This mutates the wrapped `model`! Be careful using `model` after passing it to this class.**

        Build a new ControlModel around a model instance, initializing control on
        the layers specified in `layer_ids` or `layer_zones`.

        To control layers #3 and 5, use layer_ids=[3,5].
        To control layers by their relative depth, use layer_zones=[[0.1, 0.5]] to
            control the layers with depth between 10% and 50% (left inclusive). you
            can specify multiple zones but no overlapping nor empty zones are allowed.
        """

        assert (layer_ids or layer_zones) and not (layer_ids and layer_zones), "Must supply either layer_ids or layer_zones argument"

        super().__init__()
        self.model = model

        # Get the number of layers
        layer_ids = list(range(get_num_hidden_layer(model)))
        nlayers = len(layer_ids)

        if layer_zones:
            self.layer_ids = []
            for start_zone, end_zone in layer_zones:
                assert (
                    start_zone < end_zone
                    and start_zone >= 0
                    and start_zone <= 1
                    and end_zone >= 0
                    and end_zone <= 1
                ), "wrong layer_zones format"
                if end_zone != 1.0:
                    new_layers = [
                        ilayer
                        for ilayer in layer_ids
                        if start_zone <= (ilayer / nlayers) < end_zone
                    ]
                else:  # trick to make sure to include the last layers if desired
                    new_layers = [
                        ilayer
                        for ilayer in layer_ids
                        if start_zone <= (ilayer / nlayers)
                    ]
                assert new_layers, f"No layers found in zone {start_zone} to {end_zone}"
                assert not any(nl in self.layer_ids for nl in new_layers), "Overlapping zones found"
                self.layer_ids.extend(new_layers)
        else:
            # remap to make sure they are not negative
            self.layer_ids = layer_ids

        assert self.layer_ids, "No layers to control"

        layers = model_layer_list(model)
        assert len(layers) == len(layer_ids)

        for layer_id in self.layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, ControlModule):
                layers[layer_id] = ControlModule(layer)
            else:
                warnings.warn(
                    "Trying to rewrap a wrapped model! Probably not what you want! Try calling .unwrap first."
                )

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device

    def unwrap(self) -> PreTrainedModel:
        """
        Removes the mutations done to the wrapped model and returns it.
        After using this method, `set_control` and `reset` will not work.
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layers[layer_id] = layers[layer_id].block
        return self.model

    def set_control(
        self, control: "ControlVector", coeff: float = 1.0, **kwargs
    ) -> None:
        """
        Set a `ControlVector` for the layers this ControlModel handles, with a strength given
        by `coeff`. (Negative `coeff` values invert the control vector, e.g. happiness→sadness.)
        `coeff` defaults to `1.0`.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        raw_control = {}
        for layer_id in self.layer_ids:
            raw_control[layer_id] = torch.tensor(
                coeff * control.directions[layer_id]
            ).to(self.model.device, dtype=self.model.dtype)
        self.set_raw_control(raw_control, **kwargs)

    def reset(self) -> None:
        """
        Resets the control for all layer_ids, returning the model to base behavior.
        """
        self.set_raw_control(None)

    def set_raw_control(
        self, control: dict[int, torch.Tensor] | None, **kwargs
    ) -> None:
        """
        Set or remove control parameters to the layers this ControlModel handles.
        The keys of `control` should be equal to or a superset of the `layer_ids` passed to __init__.
        Only those layers will be controlled, any others in `control` will be ignored.

        Passing `control=None` will reset the control tensor for all layer_ids, making the model act
        like a non-control model.

        Additional kwargs:
        - `normalize: bool`: track the magnitude of the non-modified activation, and rescale the
          activation to that magnitude after control (default: `False`)
        - `operator: Callable[[Tensor, Tensor], Tensor]`: how to combine the base output and control
          (default: +)
        """

        layers = model_layer_list(self.model)
        for layer_id in self.layer_ids:
            layer: ControlModule = layers[layer_id]  # type: ignore
            if control is None:
                layer.reset()
            else:
                layer.set_control(BlockControlParams(control[layer_id], **kwargs))

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@dataclasses.dataclass
class BlockControlParams:
    control: torch.Tensor | None = None
    normalize: bool = False
    operator: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )

    @classmethod
    def default(cls) -> "BlockControlParams":
        return cls()


class ControlModule(torch.nn.Module):
    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.block: torch.nn.Module = block
        self.params: BlockControlParams = BlockControlParams.default()

    def set_control(self, params: BlockControlParams) -> None:
        self.params = params

    def reset(self) -> None:
        self.set_control(BlockControlParams.default())

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        control = self.params.control

        if control is None:
            return output
        elif len(control.shape) == 1:
            control = control.reshape(1, 1, -1)

        if isinstance(output, tuple):
            modified = output[0]
        else:
            modified = output

        assert len(control.shape) == len(modified.shape)
        control = control.to(modified.device)

        if self.params.normalize:
            norm_pre = torch.norm(modified, dim=-1, keepdim=True)

        # we should ignore the padding tokens when doing the activation addition
        # mask has ones for non padding tokens and zeros at padding tokens.
        # only tested this on left padding
        if "position_ids" in kwargs:
            pos = kwargs["position_ids"]
            zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
            col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
            target_shape = modified.shape
            mask = (
                (col_indices >= zero_indices)
                .float()
                .reshape(target_shape[0], target_shape[1], 1)
            )
            mask = mask.to(modified.dtype).to(modified.device)
            modified = self.params.operator(modified, control * mask)
        else:
            modified = self.params.operator(modified, control)


        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified

        return output


def model_layer_list(model: ControlModel | PreTrainedModel) -> torch.nn.ModuleList:
    if isinstance(model, ControlModel):
        model = model.model

    if hasattr(model, "language_model"):  # gemmma3 like
        layers = model.language_model.layers
    elif hasattr(model, "layers"):  # qwen3-like
        layers = model.layers
    elif hasattr(model, "base_model"):  # mamba like
        layers = model.base_model.layers
    elif hasattr(model, "transformer"):  # gpt-2-like
        layers = model.transformer.h
    elif hasattr(model, "model"):  # mistral-like
        layers = model.model.layers
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")
