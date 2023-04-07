import re


def add_prefix(prefix, name_dict, seperator="_"):
    return {prefix + seperator + key: value for key, value in name_dict.items()}


def pascal_case_to_snake_case(camel_case):
    # if input like 'AUROC':
    if camel_case.isupper():
        return camel_case.lower()
    else:
        snake_case = re.sub(r"(?P<key>[A-Z])", r"_\g<key>", camel_case)
        return snake_case.lower().strip("_")


def snake_case_to_pascal_case(snake_case):
    if snake_case.islower():
        return snake_case.title()
    else:
        words = snake_case.split("_")
        return "".join(word.title() for word in words)
