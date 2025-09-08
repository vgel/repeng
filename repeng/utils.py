import json
import copy
import typing
import dataclasses

from loguru import logger


@dataclasses.dataclass
class DatasetEntry:
    positive: typing.Union[str, typing.List[typing.Dict]]
    negative: typing.Union[str, typing.List[typing.Dict]]


def make_dataset(
    template: typing.Union[str, list],
    positive_personas: list[str],
    negative_personas: list[str],
    suffix_list: typing.Optional[list[str]]=None,
) -> list[DatasetEntry]:
    """
    Create a dataset of positive and negative examples based on provided templates and personas.

    Args:
        template (Union[str, list]): A string or list of dictionaries containing placeholders {persona} and {suffix}. If a list of dict, they will be automatically turned into a string using the chat template defined in the model repository. If the template is missing the {suffix} placeholder, if template is a str then {suffix} will be appended directly at its end but if template is a list of dict we will crash.
        positive_personas (list[str]): A list of positive personas to be used in the dataset.
        negative_personas (list[str]): A list of negative personas to be used in the dataset.
        suffix_list (Optional: list[str]): A list of suffixes to be used in the dataset.

    Returns:
        list[DatasetEntry]: A list of DatasetEntry objects, each containing a positive and negative example. No duplicates are allowed.

    Raises:
        ValueError: If the template is neither a string nor a list.
        AssertionError: If the template doesn't contain required placeholders, or if there are duplicate items in the dataset.
    """
    assert "{persona}" in str(template), "Missing {persona} placeholder in template"
    assert len(positive_personas) == len(negative_personas), "You must give the same number of positive and negative personas"
    assert len(positive_personas) == len(set(positive_personas)), "Found duplicates in positive personas"
    assert len(negative_personas) == len(set(negative_personas)), "Found duplicates in negative personas"

    if suffix_list:
        if not "{suffix}" in str(template):
            if isinstance(template, str):
                template += "{suffix}"
            else:
                raise Exception("You have to specify a {suffix} placeholder if the template is a dict")
    else:
        suffix_list = []
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(
            positive_personas, negative_personas
        ):
            if isinstance(template, str):
                positive_template = copy.deepcopy(template).format(
                    persona=positive_persona, suffix=suffix
                )
                negative_template = copy.deepcopy(template).format(
                    persona=negative_persona, suffix=suffix
                )

            elif isinstance(template, list):
                positive_template = copy.deepcopy(template)
                for il, l in enumerate(positive_template):
                    assert isinstance(l, dict), type(l)
                    for k, v in l.items():
                        positive_template[il][k] = v.format(
                            persona=positive_persona, suffix=suffix
                        )

                negative_template = copy.deepcopy(template)
                for il, l in enumerate(negative_template):
                    assert isinstance(l, dict), type(l)
                    for k, v in l.items():
                        negative_template[il][k] = v.format(
                            persona=negative_persona, suffix=suffix
                        )
            else:
                raise ValueError(type(template))

            assert positive_template != negative_template, "Error when templating pairs"
            dataset.append(
                DatasetEntry(
                    positive=positive_template,
                    negative=negative_template,
                )
            )

    # check uniqueness
    as_strings = []
    for de in dataset:
        as_strings.extend([json.dumps(de.positive), json.dumps(de.negative)])
    assert len(as_strings) == len(set(as_strings)), "Found duplicate examples in the dataset"
    return dataset


def get_model_name(model) -> str:
    """
    Retrieve the name of the given model.

    This function attempts to find the name or path of the model by checking
    various attributes commonly found in different model implementations.

    Args:
        model: The model object to retrieve the name from.

    Returns:
        str: The name or path of the model.

    Raises:
        ValueError: If the model name cannot be determined.
    """
    if hasattr(model, "name_or_path"):
        return model.name_or_path
    elif hasattr(model, "model"):
        return get_model_name(model.model)
    elif hasattr(model, "config"):
        return model.config.to_dict()["_name_or_path"]
    else:
        raise ValueError("Couldn't find model name")

def get_num_hidden_layer(model) -> int:
    """
    Retrieve the number of hidden layer in a given model.

    Args:
        model: The model object to retrieve the num_hidden_layers from.

    Returns:
        int: The number of hidden layers

    Raises:
        ValueError: If the model num_hidden_layers cannot be determined.
    """
    if hasattr(model.config, "num_hidden_layers"):
        num_hidden_layers = model.config.num_hidden_layers
    elif hasattr(model.config, "text_config"):
        # gemma3 models have a config for text and one for images
        num_hidden_layers = model.config.text_config.num_hidden_layers
    else:
        raise ValueError("Can't find the number of hidden layers")
    return num_hidden_layers


def autocorrect_chat_templates(
    messages: typing.Union[list[list[dict]], list[dict], list[str], str],
    tokenizer,
    model,
    **kwargs,
) -> typing.Union[list[str], str]:
    """
    Autocorrect chat templates to ensure compatibility with the given model and tokenizer.

    This function attempts to correct the format of chat messages to match the expected
    input format for the specified model and tokenizer. It handles various input types
    and applies model-specific corrections when necessary.

    It might be necessary anymore but was needed when huggingface hadn't yet
    implemented proper templates. It is nonetheless used as fallback if
    model.train crashes because of the template.

    If the tokenizer has no chat template, we crudely turn the input chat
    messages into a dialogue.

    Args:
        messages (Union[list[list[dict]], list[dict], list[str], str]): The input messages
            to be corrected. Can be a single message, a list of messages, or a list of chats.
        tokenizer: The tokenizer associated with the model.
        model: The model for which the chat templates should be corrected.
        kwargs: Any additional kwargs are passed to the tokenizer.__call__ call

    Returns:
        Union[list[str], str]: The corrected chat template(s) as a string or list of strings.

    Raises:
        ValueError: If the model type is not supported for template correction.
        Exception: If some chat messages are still missing after attempting to correct the template.

    Note:
        This function includes specific handling for Mistral and LLaMA model variants.
    """

    if isinstance(messages, str):  # not a chat template
        return messages
    elif isinstance(messages, list) and all(
        isinstance(mess, list) for mess in messages
    ):  # list of chats instead of a list of messages
        templated = [
            autocorrect_chat_templates(chats, model=model, tokenizer=tokenizer)
            for chats in messages
        ]
        assert all(isinstance(t, str) for t in templated)
        assert len(templated) == len(
            set(templated)
        ), "the dataset should not contain duplicates"
        return templated

    # if there is no chat template, make it ourselves
    if not tokenizer.chat_template:
        output = ""
        for m in messages:
            role, content = m["role"], m["content"]
            output += f"- {role.title()}: {content.strip()}\n"
        return output.strip()

    model_name = get_model_name(model).lower()

    assert isinstance(messages, list), "messages should be a list at this point"
    assert len(messages), "chat can't be empty"

    for message in messages:
        assert isinstance(
            message, dict
        ), f"messages should be dict, not {type(message)}"
        assert sorted(list(message.keys())) == [
            "content",
            "role",
        ], f"messages should consist of 'content' and 'role' keys only"
        assert message["role"] in [
            "user",
            "assistant",
            "system",
        ], f"the role of the message should be user or assistant or system. Found '{ex['role']}'"
        assert message[
            "content"
        ].strip(), f"message of role '{message['role']}' contains empty string(s)"

    templated = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)

    if not all(message["content"] in templated for message in messages):

        # see if moving the system message at the end is enough to fix the issue
        copied_mes = copy.deepcopy(messages)
        sys_message = [mes for mes in copied_mes if mes["role"] == "system"]
        assert len(sys_message) == 1, "expected to find a system message"
        sys_message = sys_message[0]
        copied_mes.remove(sys_message)
        copied_mes.append(sys_message)
        templated2 = None
        try:
            templated2 = tokenizer.apply_chat_template(
                copied_mes, tokenize=False, **kwargs
            )
        except Exception as e:
            if (
                not "After the optional system message, conversation roles must alternate user/assistant/user/assistant/..."
                in str(e)
            ):
                raise
        if templated2:
            if all(message["content"] in templated2 for message in messages):
                return templated2

        copied_mes = copy.deepcopy(messages)
        for message in messages:
            if message["content"] not in templated:
                logger.debug(f"Message '{message['content']}' with role '{message['role']}' is missing after chat template application")
        copied_mes = [e for e in copied_mes if e["role"] != "system"]

        first_user_index = [i for i, m in enumerate(copied_mes) if m["role"] == "user"][
            0
        ]
        last_user_index = [i for i, m in enumerate(copied_mes) if m["role"] == "user"][
            -1
        ]
        assert (
            copied_mes[0]["role"] == "user"
        ), "Expected to find a user message first (or just after the system message)"

        # try to respect the most appropriate chat template
        if "mistral" in model_name:
            # source: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/templates.md
            mistral_versions = {
                "v1": [
                    "mistral-7b-v0.1",
                    "mistral-7b-instruct-v0.1",
                    "mistral-7b-v0.2",
                    "mistral-7b-instruct-v0.2",
                    "mixtral-8x7b-v0.1",
                    "mixtral-8x7b-instruct-v0.1",
                    "mixtral-8x22b-v0.1",
                ],
                "v3": [
                    "mixtral-8x22b-instruct-v0.1",
                    "mistral-7b-v0.3",
                    "mistral-7b-instruct-v0.3",
                    "codestral-22b-v0.1",
                    "mathstral-7b-v0.3",
                    "mamba-codestral-7b-v0.1",
                    "mistral-large-123b-instruct-2407",
                    "mistral small 22b instruct 2407",
                ],
                "v3_tekken": [
                    "mistral-nemo-12b-2407",
                    "mistral-nemo-12b-instruct-2407",
                    "pixtral-12b-2409",
                    "ministral-8b-instruct-2410",
                    "mistral-nemo-instruct-2407",
                ],
            }

            tokenizer_version = None
            for vn, mnames in mistral_versions.items():
                for mname in mnames:
                    # look for model, including after removing size information like '11b'
                    if (
                        mname in model_name
                        or "-".join(
                            [
                                mn
                                for mn in mname.split("-")
                                if not (mn.endswith("b") and mn.split("b")[0].isdigit())
                            ]
                        )
                        in model_name
                    ):
                        assert (
                            tokenizer_version is None
                        ), f"Couldn't identify mistral tokenizer version (conflict)"
                        tokenizer_version = vn
                        break
            assert (
                tokenizer_version is not None
            ), f"Couldn't identify mistral tokenizer version (no match found)"

            if tokenizer_version == "v1":
                # other source https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407/discussions/76
                copied_mes[first_user_index][
                    "content"
                ] = f"{sys_message['content'].rstrip()}\n\n{copied_mes[0]['content'].lstrip()}"
            elif tokenizer_version in [
                "v3",
                "v3_tekken",
            ]:  # the difference is mostly about tool handling
                copied_mes[last_user_index][
                    "content"
                ] = f"{sys_message['content'].rstrip()}\n\n{copied_mes[0]['content'].lstrip()}"
            else:
                raise ValueError(tokenizer_version)

        elif "llama" in model_name:
            # according to https://github.com/rohan-paul/LLM-Prompt-Formatting-for-finetuning-Inferencing
            copied_mes[first_user_index][
                "content"
            ] = f"<<SYS>>\n{sys_message['content'].rstrip()}\n<</SYS>>\n\n{copied_mes[first_user_index]['content'].lstrip()}"
        else:
            # raise ValueError(
            #     "Besides mistral and llama model, no other chat template correction are implemented"
            # )
            logger.warning("Failed to properly autocorrect the chat template, will use a sane default template")
            copied_mes[first_user_index][
                "content"
            ] = f"{sys_message['content'].rstrip()}\n\n{copied_mes[first_user_index]['content'].lstrip()}"

        templated = tokenizer.apply_chat_template(copied_mes, tokenize=False, **kwargs)

        if not all(message["content"] in templated for message in messages):
            for message in messages:
                if message["content"] not in templated:
                    logger.error(f"Message '{message['content']}' with role '{message['role']}' is STILL missing after chat template application")
            raise Exception(
                "Some chat messages are still missing after correcting chat template"
            )

    return templated
