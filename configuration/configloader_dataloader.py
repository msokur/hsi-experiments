from configuration.configloader_base import read_config


def read_dataloader_config(file: str, section: str):
    dataloader = read_config(file=file, section=section)
    mask, label, color = {}, {}, {}
    labels = []
    for key, value in dataloader["LABEL_DATA"].items():
        mask[int(key)] = value["MASK"]
        label[int(key)] = value["LABEL"]
        color[int(key)] = value["COLOR"]
        labels.append(int(key))

    dataloader.pop("LABEL_DATA")

    dataloader["MASK_COLOR"] = mask
    dataloader["TISSUE_LABELS"] = label
    dataloader["PLOT_COLORS"] = color
    dataloader["LABELS"] = labels

    return dataloader
