import abc
import warnings
from shutil import rmtree
from typing import List, Dict, Iterable

import numpy as np
import platform
import os


class DataArchive:
    def __init__(self):
        self.system = platform.system()

    @staticmethod
    @abc.abstractmethod
    def get_path(file) -> str:
        """
        Returns the path from the data archive file.

        :param file: the archive file

        :return: A string with the file path
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_paths(archive_path: str) -> List[str]:
        """
        Returns a list with all group paths in the giving path.

        :param archive_path: String with the data archive groups

        :return: A list with the absolute path from every group
        """
        pass

    def get_name(self, path: str) -> str:
        """
        Returns the name from the group.

        :param path: Group path

        :return: Group name
        """
        # check if Windows or Linux/Mac dir
        if self.system == "Windows":
            name = os.path.split(path)[-1]
        else:
            name = path
            if "\\" in name:
                name = path.split("\\")[-1]
            if "/" in name:
                name = path.split("/")[-1]
        # remove data extension
        if "." in name:
            return name.split(".")[0]
        else:
            return name

    @abc.abstractmethod
    def all_data_generator(self, archive_path: str) -> Iterable:
        """
        Get a generator with all Groups in the data archive.

        :param archive_path: The path with the data to load

        :return: A Generator with a loaded group
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_datas(data_path: str):
        """
        Loads a group with all array.

        :param data_path: The path from the group

        :return: Data Archive file
        """
        pass

    @abc.abstractmethod
    def get_data(self, data_path: str, data_name: str) -> np.ndarray:
        """
        Loads a specific array in the group.

        :param data_path: Path from the group to load
        :param data_name: The name from the array to load

        :return: The array as numpy array
        """
        pass

    @abc.abstractmethod
    def save_group(self, save_path: str, group_name: str, datas: Dict[str, np.ndarray]) -> None:
        """
        Save a group with different array inside. Every key in the datas dictionary is an array name

        :param save_path: Path for the file to save
        :param group_name: File name
        :param datas: Dictionary with array name and values
        """
        pass

    @abc.abstractmethod
    def save_data(self, save_path: str, data_name: str, data: np.ndarray) -> None:
        """
        Save an array in a group and create a new group if the group not exist. Overwrite the array if it's exist.

        :param save_path: File path with file name
        :param data_name: Array name
        :param data: Array to save
        """
        pass

    @abc.abstractmethod
    def append_data(self, file_path: str, append_datas: Dict[str, np.ndarray]) -> None:
        """
        Appends datas to a existing data archive.

        :param file_path: The path to the file to append
        :param append_datas: The data to append

        :raises ValueError: When the append data has not all data from the data archive
        :raises FileNorFoundError: If the file to append not exist
        :raises UserWarning: If more data in the append data then in the data archive
        """
        pass

    @staticmethod
    def delete_archive(delete_path: str):
        """
        Delete every archive file in the given path.

        :param delete_path: Archive path to delete
        """
        print(f"--- Delete archive in {delete_path}. ---")
        rmtree(delete_path)
        print("--- Delete finish ---")

    @staticmethod
    def _file_not_found(file: str):
        raise FileNotFoundError(f"File: '{file}' not found to append data!")

    @staticmethod
    def _check_keys(name: str, base_keys: set, append_keys: set):
        a = base_keys - append_keys
        b = append_keys - base_keys

        if a.__len__() > 0:
            raise ValueError(f"Different keys in data archive '{name}' to combine data!")
        elif b.__len__() > 0:
            not_in_data = ", ".join(b)
            raise warnings.warn(f"Data '{not_in_data}' will be not append to data archive!")
