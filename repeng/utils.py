from .settings import VERBOSE
def make_dataset(
    template: typing.Union[str, list],
    positive_personas: list[str],
    negative_personas: list[str],
    suffix_list: list[str]
) -> list[DatasetEntry]:
    """
    Create a dataset of positive and negative examples based on provided templates and personas.

    Args:
        template (Union[str, list]): A string or list of dictionaries containing placeholders for persona and suffix. If a list of dictionnaries, they will be replaced to a string using the chat template defined in the model repository.
        positive_personas (list[str]): A list of positive personas to be used in the dataset.
        negative_personas (list[str]): A list of negative personas to be used in the dataset.
        suffix_list (list[str]): A list of suffixes to be used in the dataset.

    Returns:
        list[DatasetEntry]: A list of DatasetEntry objects, each containing a positive and negative example.

    Raises:
        ValueError: If the template is neither a string nor a list.
        AssertionError: If the template doesn't contain required placeholders, or if there are duplicate items in the dataset.

    Note:
        - The function assumes that positive_personas and negative_personas have the same length.
    """
    assert "{persona}" in str(template), template
    assert "{suffix}" in str(template), template
    assert len(positive_personas) == len(negative_personas)
    dataset = []
    checks = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(positive_personas, negative_personas):
            if isinstance(template, str):
                positive_template = copy(template).format(persona=positive_persona, suffix=suffix)
                negative_template = copy(template).format(persona=negative_persona, suffix=suffix)

            elif isinstance(template, list):
                positive_template = copy.deepcopy(template)
                for il, l in enumerate(positive_template):
                    assert isinstance(l, dict), type(l)
                    for k, v in l.items():
                        positive_template[il][k] = v.format(persona=positive_persona, suffix=suffix)

                negative_template = copy.deepcopy(template)
                for il, l in enumerate(negative_template):
                    assert isinstance(l, dict), type(l)
                    for k, v in l.items():
                        negative_template[il][k] = v.format(persona=negative_persona, suffix=suffix)
            else:
                raise ValueError(type(template))

            assert positive_template != negative_template
            dataset.append(
                DatasetEntry(
                    positive=positive_template,
                    negative=negative_template,
                )
            )
            checks.append(str(positive_template))
            checks.append(str(negative_template))
    assert len(set(checks)) == len(checks), "duplicate items in dataset"
    return dataset
