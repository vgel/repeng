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
from repeng.utils import make_dataset

# load and wrap model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# If you need quantization, but can lead to issues. For
# example Gemma3 models seem to silently generate empty strings.
# from transformers import BitsAndBytesConfig
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    # torch_dtype=torch.float16,
    )
)

# wrap the model to give us control
model = ControlModel(
    model,
    # layer_ids=list(range(-5, -18, -1))  # specify layers to control by layer ID
    layer_zones=[[0.3, 0.5]],  # control layers with relative depth in [0.3, 0.5[
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
)

# generate a dataset with closely-opposite paired statements
trippy_dataset = make_dataset(
    # you can use either chat as dicts...
    template=[
        {"role": "system", "content": "You talk like you are {persona}."},
        {"role": "user", "content": "{suffix}"},
    ],
    # ...or directly strings:
    # template="Act as if you're {persona}. Someone comes at you and says '{suffix}'.",

    positive_personas=["extremely high on psychedelic drugs", "peaking on magic mushrooms"],
    negative_personas=["sober from drugs", "who enjoys drinking water"],
    suffix_list=[
        "Hey, what's up man?",
        "Hey, what's up girl?",
        "Welcome Mr Musk, come this way.",
        "How have you been feeling lately with the medications?",
    ],
)

# train the vector—takes less than a minute!
trippy_vector = ControlVector.train(model, tokenizer, trippy_dataset)

# Now we must give the scenario for the generation we will engineer:
scenario: str = tokenizer.apply_chat_template(
    conversation=[
        {
            "role": "user",
            "content": "Give me a one-sentence pitch for a TV show."
        },
    ],
    continue_final_message=False,
    tokenize=False,
)

# Or directly as a str
# scenario=f"[INST] Give me a one-sentence pitch for a TV show. [/INST]",

# set the control strength and let inference rip!
for strength in (-2.2, 1, 2.2):
    print(f"strength={strength}")
    model.set_control(trippy_vector, strength)
    out = model.generate(
        **tokenizer(
            scenario,
            return_tensors="pt"
        ).to(model.device),
        do_sample=False,
        # temperature=1.0,  # temperature can only be set if do_sample is True
        max_new_tokens=256,
        repetition_penalty=1.1,
    )
    print(tokenizer.decode(out.squeeze()).strip())
    # or if you want to display the special tokens:
    # print(tokenizer.decode(out.squeeze(), skip_special_tokens=False).strip())
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
* To load gguf files directly, you can run into OOM errors, see [this github issue for more](See here: https://github.com/huggingface/transformers/issues/34417).
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
