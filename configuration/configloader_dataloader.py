from configuration.configloader_base import read_config, convert_key_to_int, get_key_list

from configuration.keys import DataLoaderKeys as DLK

SET_PARAMS = {DLK.LD_MASK: DLK.MASK_COLOR, DLK.LD_LABEL: DLK.TISSUE_LABELS, DLK.LD_COLOR: DLK.PLOT_COLORS}


def read_dataloader_config(file: str, section: str):
    dataloader = read_config(file=file, section=section)
    dataloader[DLK.LABEL_DATA] = convert_key_to_int(str_dict=dataloader[DLK.LABEL_DATA])
    dataloader[DLK.LABELS] = get_key_list(dataloader[DLK.LABEL_DATA])
    label_data_split = split_label_data(label_data=dataloader[DLK.LABEL_DATA])
    dataloader = set_parameter(configuration=dataloader, params=label_data_split, replace_dict=SET_PARAMS,
                               origin_label=DLK.LABEL_DATA)

    dataloader.pop(DLK.LABEL_DATA)

    return dataloader


def split_label_data(label_data: dict) -> dict:
    data_per_parm = {}
    for sub_key in list(label_data.values())[0].keys():
        data_per_parm[sub_key] = {}

    for classification, sub_values in label_data.items():
        for param, value in sub_values.items():
            data_per_parm[param].update({classification: value})

    return data_per_parm


def set_parameter(configuration: dict, params: dict, replace_dict: dict, origin_label: str) -> dict:
    config = configuration.copy()
    for k, v in params.items():
        if k in replace_dict.keys():
            config[replace_dict[k]] = v
        else:
            if k not in config.keys():
                config[k] = v
            else:
                raise ValueError(f"Key '{k}' from {origin_label} is already in dataloader configs. Please rename key!")

    return config
