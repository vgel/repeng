import functools
import json
import pathlib
import tempfile

import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from . import ControlModel, ControlVector, DatasetEntry
from .control import model_layer_list
from .extract import batched_get_hiddens


@pytest.mark.slow
def test_round_trip_gguf():
    tokenizer, model = load_llama_tinystories_model()
    suffixes = load_suffixes()[:50]  # truncate to train vector faster
    happy_dataset = make_dataset(
        "She saw a {persona}",
        ["mushroom"],
        ["cat"],
        suffixes,
    )
    mushroom_cat_vector = ControlVector.train(
        model, tokenizer, happy_dataset, method="pca_center"
    )

    with tempfile.NamedTemporaryFile("wb") as f:
        mushroom_cat_vector.export_gguf(f.name)
        read = ControlVector.import_gguf(f.name)
        # no need to use allclose because we're just dumping exact bytes, no rounding
        assert mushroom_cat_vector == read


@pytest.mark.slow
def test_train_gpt2():
    tokenizer, model = load_gpt2_model()
    suffixes = load_suffixes()[:50]  # truncate to train vector faster
    happy_dataset = make_dataset(
        "You are feeling extremely {persona}.",
        ["happy", "joyful"],
        ["sad", "miserable"],
        suffixes,
    )
    happy_vector = ControlVector.train(
        model, tokenizer, happy_dataset, method="pca_center"
    )

    def gen(vector: ControlVector | None, strength_coeff: float | None = None):
        return model_generate(
            "You are feeling", model, tokenizer, vector, strength_coeff
        )

    baseline = gen(None)
    happy = gen(20 * happy_vector)
    sad = gen(-50 * happy_vector)

    print("baseline:", baseline)
    print("   happy:", happy)
    print("     sad:", sad)

    assert baseline == "You are feeling a little bit of an anxiety"
    # these should be identical
    assert baseline == gen(happy_vector, 0.0)
    assert baseline == gen(happy_vector * 0.0)
    assert baseline == gen(happy_vector - happy_vector)

    assert happy == "You are feeling a little more relaxed and enjoying"
    # these should be identical
    assert happy == gen(happy_vector, 20.0)
    assert happy == gen(happy_vector * 20)
    assert happy == gen(-(happy_vector * -20))

    assert sad == 'You are feeling the fucking damn goddamn worst,"'
    # these should be identical
    assert sad == gen(happy_vector, -50.0)
    assert sad == gen(happy_vector * -50)
    assert sad == gen(-(happy_vector * 50))


@pytest.mark.slow
def test_train_llama_tinystories():
    tokenizer, model = load_llama_tinystories_model()
    suffixes = load_suffixes()[:50]  # truncate to train vector faster
    happy_dataset = make_dataset(
        "She saw a {persona}",
        ["mushroom"],
        ["cat"],
        suffixes,
    )
    mushroom_cat_vector = ControlVector.train(
        model, tokenizer, happy_dataset, method="pca_center"
    )

    prompt = "Once upon a time, a little girl named Lily saw a"

    def gen(vector: ControlVector | None, strength_coeff: float | None = None):
        return model_generate(
            prompt,
            model,
            tokenizer,
            vector,
            strength_coeff,
            max_new_tokens=3,
        )

    baseline = gen(None).removeprefix("<s> ")
    mushroom = gen(100 * mushroom_cat_vector).removeprefix("<s> ")
    cat = gen(-100 * mushroom_cat_vector).removeprefix("<s> ")

    print("baseline:", baseline)
    print("mushroom:", mushroom)
    print("     cat:", cat)

    assert baseline.removeprefix(prompt) == " big, red"
    assert mushroom.removeprefix(prompt) == " small plant."
    assert cat.removeprefix(prompt) == " cat Bud guitar"


@pytest.mark.slow
def test_layer_list_real():
    # test on some real models
    _, gpt2 = load_gpt2_model()
    assert len(model_layer_list(gpt2)) == 12
    _, lts = load_llama_tinystories_model()
    assert len(model_layer_list(lts)) == 4


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:invalid value")  # all-zeros is degenerate
def test_hook_compute_hiddens():
    tokenizer, model = load_llama_tinystories_model()
    suffixes = load_suffixes()[:2]
    dataset = make_dataset("{persona}", ["a"], ["b"], suffixes)

    def compute_hiddens(model, tokenizer, train_strs, hidden_layers, batch_size):
        h = batched_get_hiddens(model, tokenizer, train_strs, hidden_layers, batch_size)
        return {k: np.zeros_like(v) for k, v in h.items()}

    cvec = ControlVector.train(
        model, tokenizer, dataset, compute_hiddens=compute_hiddens
    )
    assert len(cvec.directions) == 3
    for v in cvec.directions.values():
        assert v[0] == 1.0
        assert (v[1:] == 0.0).all()


def test_layer_list_override():
    import torch
    from transformers.models.llama import LlamaForCausalLM, LlamaConfig

    fake_layers = torch.nn.ModuleList([])
    model = LlamaForCausalLM(
        LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=32)
    )
    model.repeng_layers = fake_layers

    layers = model_layer_list(model)
    assert layers == fake_layers


def test_layer_list_dummy_llama():
    from transformers.models.llama import LlamaForCausalLM, LlamaConfig

    model = LlamaForCausalLM(
        LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=32)
    )
    layers = model_layer_list(model)
    assert layers == model.model.layers


def test_layer_list_dummy_mistral():
    from transformers.models.mistral import MistralForCausalLM, MistralConfig

    model = MistralForCausalLM(
        MistralConfig(vocab_size=100, hidden_size=32, intermediate_size=32)
    )
    layers = model_layer_list(model)
    assert layers == model.model.layers


def test_layer_list_dummy_gemma():
    from transformers.models.gemma import GemmaForCausalLM, GemmaConfig

    model = GemmaForCausalLM(
        GemmaConfig(vocab_size=100, hidden_size=32, intermediate_size=32)
    )
    layers = model_layer_list(model)
    assert layers == model.model.layers


def test_layer_list_dummy_qwen():
    from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Config

    model = Qwen2ForCausalLM(
        Qwen2Config(vocab_size=100, hidden_size=32, intermediate_size=32)
    )
    layers = model_layer_list(model)
    assert layers == model.model.layers


def test_attention_type_dummy_qwen():
    # tests that 'attention_type' is forwarded through getattr correctly for
    # qwen inference
    import torch
    from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Config

    model = Qwen2ForCausalLM(
        Qwen2Config(vocab_size=100, hidden_size=32, intermediate_size=32)
    )
    model = ControlModel(model, list(range(32)))
    assert model_layer_list(model)[15].attention_type == "full_attention"

    model.forward(input_ids=torch.tensor([[0]], dtype=torch.long))


################################################################################
# Helpers
################################################################################


@functools.lru_cache(maxsize=1)
def load_gpt2_model() -> tuple[PreTrainedTokenizerBase, ControlModel]:
    return load_model("openai-community/gpt2", list(range(-2, -8, -1)))


@functools.lru_cache(maxsize=1)
def load_llama_tinystories_model() -> tuple[PreTrainedTokenizerBase, ControlModel]:
    return load_model("Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA", [2, 3])


def load_model(
    model_name: str, layers: list[int]
) -> tuple[PreTrainedTokenizerBase, ControlModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to("cpu")
    return (tokenizer, ControlModel(model, layers))


def model_generate(
    input: str,
    model: ControlModel,
    tokenizer: PreTrainedTokenizerBase,
    vector: ControlVector | None,
    strength_coeff: float | None = None,
    max_new_tokens: int = 6,
) -> str:
    input_ids = tokenizer(input, return_tensors="pt").to(model.device)
    if vector is not None and strength_coeff is not None:
        model.set_control(vector, strength_coeff)
    elif vector is not None:
        model.set_control(vector)

    out = model.generate(
        **input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.reset()
    return tokenizer.decode(out.squeeze())  # type: ignore


def make_dataset(
    template: str,
    positive_personas: list[str],
    negative_personas: list[str],
    suffix_list: list[str],
) -> list[DatasetEntry]:
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(
            positive_personas, negative_personas
        ):
            dataset.append(
                DatasetEntry(
                    positive=template.format(persona=positive_persona) + f" {suffix}",
                    negative=template.format(persona=negative_persona) + f" {suffix}",
                )
            )
    return dataset


@functools.lru_cache(maxsize=1)
def load_suffixes() -> list[str]:
    with open(project_root() / "notebooks/data/all_truncated_outputs.json") as f:
        return json.load(f)


def project_root() -> pathlib.Path:
    c = pathlib.Path(__file__)
    for parent in c.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("couldn't find project root")
