def add_prefix(prefix, name_dict, seperator='_'):
    return {prefix+seperator+key: value for key, value in name_dict.items()}