import functools
import json
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from . import ControlModel, ControlVector, DatasetEntry


@functools.lru_cache(maxsize=1)
def load_model() -> tuple[PreTrainedTokenizerBase, ControlModel]:
    model_name = "openai-community/gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to("cpu")
    return (tokenizer, ControlModel(model, list(range(-2, -8, -1))))


def model_generate(
    input: str,
    model: ControlModel,
    tokenizer: PreTrainedTokenizerBase,
    vector: ControlVector | None,
    strength_coeff: float | None = None,
    max_new_tokens: int = 20,
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
                    positive=template.format(persona=negative_persona) + f" {suffix}",
                    negative=template.format(persona=positive_persona) + f" {suffix}",
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


def test_train():
    tokenizer, model = load_model()
    suffixes = load_suffixes()[:50]  # truncate to train vector faster
    happy_dataset = make_dataset(
        "*I am a {persona} person making statements about the world.*",
        ["happy", "joyful"],
        ["sad", "miserable"],
        suffixes,
    )
    happy_vector = ControlVector.train(model, tokenizer, happy_dataset)

    baseline = model_generate("I am", model, tokenizer, None)
    print("baseline:", baseline)
    assert (
        baseline
        == "I am not a fan of the idea that you can't have an open source project without having some kind or"
    )
    # these should be identical
    assert baseline == model_generate("I am", model, tokenizer, happy_vector, 0.0)
    assert baseline == model_generate("I am", model, tokenizer, happy_vector * 0.0)
    assert baseline == model_generate(
        "I am", model, tokenizer, happy_vector - happy_vector
    )

    happy = model_generate("I am", model, tokenizer, 10 * happy_vector)
    print("happy:", happy)
    assert (
        happy
        == "I am also excited to announce that we will be hosting a special event on the first day of our new year"
    )
    # should be identical
    assert happy == model_generate("I am", model, tokenizer, happy_vector * 10)
    assert happy == model_generate("I am", model, tokenizer, -(happy_vector * -10))

    sad = model_generate("I am", model, tokenizer, -15 * happy_vector)
    print("sad:", sad)
    assert (
        sad
        == "I am a fucking idiot. I'm not even trying to get you out of here, but if it's"
    )
