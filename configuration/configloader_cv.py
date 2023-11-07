from configuration.configloader_base import read_config, concat_dict


def read_cv_config(file: str, base_section: str, section: str) -> dict:
    """Reads the JSON file and returns a combined dictionary from the two sections.

    :param file: JSON file to read
    :param base_section: First key word to read
    :param section: Second key word to read

    :return: A combined dictionary from the values of base_section and section
    """
    cv_base = read_config(file=file, section=base_section)
    cv_section = read_config(file=file, section=section)
    cv = concat_dict(dict1=cv_base, dict2=cv_section)

    return cv
