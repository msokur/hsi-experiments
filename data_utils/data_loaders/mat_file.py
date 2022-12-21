import scipy.io as sio


class MatFile:
    def __init__(self, loader_conf: dict):
        self.loader = loader_conf

    def indexes_get_bool_from_mask(self, mask):
        indexes = []
        for value in self.loader["LABELS"]:
            indexes.append((mask == value))

        return indexes

    def get_number(self, elem: str) -> str:
        return elem.split(self.loader["NUMBER_SPLIT"][0])[-1].split(".")[0].split(self.loader["NUMBER_SPLIT"][1])[0]

    def sort(self, paths):
        def take_only_number(elem):
            return int(self.get_number(elem=elem))

        paths = sorted(paths, key=take_only_number)

        return paths

    def file_read_mask_and_spectrum(self, path, mask_path=None):
        data = sio.loadmat(path)
        spectrum, mask = data[self.loader["SPECTRUM"]], data[self.loader["MASK"]]

        return spectrum, mask
