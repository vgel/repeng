# repeng

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/vgel/repeng/ci.yml?label=ci)](https://github.com/vgel/repeng/actions)
[![PyPI - Version](https://img.shields.io/pypi/v/repeng)](https://pypi.org/project/repeng/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/repeng)](https://pypi.org/project/repeng/)
[![GitHub License](https://img.shields.io/github/license/vgel/repeng)](https://github.com/vgel/repeng/blob/main/LICENSE)

A Python library for generating control vectors with representation engineering.
Train a vector in less than sixty seconds!

_For a full example, see the notebooks folder or [the blog post](https://vgel.me/posts/representation-engineering)._

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlVector, ControlModel, DatasetEntry

# load and wrap Mistral-7B
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = ControlModel(model, list(range(-5, -18, -1)))

def make_dataset(template: str, pos_personas: list[str], neg_personas: list[str], suffixes: list[str]):
    # see notebooks/experiments.ipynb for a definition of `make_dataset`
    ...

# generate a dataset with closely-opposite paired statements
trippy_dataset = make_dataset(
    "Act as if you're extremely {persona}.",
    ["high on psychedelic drugs"],
    ["sober from psychedelic drugs"],
    truncated_output_suffixes,
)

# train the vector—takes less than a minute!
trippy_vector = ControlVector.train(model, tokenizer, trippy_dataset)

# set the control strength and let inference rip!
for strength in (-2.2, 1, 2.2):
    print(f"strength={strength}")
    model.set_control(trippy_vector, strength)
    out = model.generate(
        **tokenizer(
            f"[INST] Give me a one-sentence pitch for a TV show. [/INST]",
            return_tensors="pt"
        ),
        do_sample=False,
        max_new_tokens=128,
        repetition_penalty=1.1,
    )
    print(tokenizer.decode(out.squeeze()).strip())
    print()
```

> strength=-2.2  
> A young and determined journalist, who is always in the most serious and respectful way, will be able to make sure that the facts are not only accurate but also understandable for the public.
>
> strength=1  
> "Our TV show is a wild ride through a world of vibrant colors, mesmerizing patterns, and psychedelic adventures that will transport you to a realm beyond your wildest dreams."
>
> strength=2.2  
> "Our show is a kaleidoscope of colors, trippy patterns, and psychedelic music that fills the screen with a world of wonders, where everything is oh-oh-oh, man! ��psy����������oodle����psy��oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

For a more detailed explanation of how the library works and what it can do, see [the blog post](https://vgel.me/posts/representation-engineering).

## Notes

* For a list of changes by version, see the [CHANGELOG](https://github.com/vgel/repeng/blob/main/CHANGELOG).
* For quantized use, you may be interested in [llama.cpp#5970](https://github.com/ggerganov/llama.cpp/pull/5970)—after training a vector with `repeng`, export it by calling `vector.export_gguf(filename)` and then use it in `llama.cpp` with any quant!
* Vector training *currently does not work* with MoE models (such as Mixtral). (This is theoretically fixable with some work, let me know if you're interested.)
* Some example notebooks require `accelerate`, which must be manually installed with `pip install accelerate`. (This can also be done in the notebook with the IPython magic `%pip install accelerate`.)

## Notice

Some of the code in this repository derives from [andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering) (MIT license).

## Citation

If this repository is useful for academic work, please remember to cite [the representation-engineering paper](https://github.com/andyzoujm/representation-engineering?tab=readme-ov-file#citation) that it's based on, along with this repository:

```
@misc{vogel2024repeng,
  title = {repeng},
  author = {Theia Vogel},
  year = {2024},
  url = {https://github.com/vgel/repeng/}
}
```
