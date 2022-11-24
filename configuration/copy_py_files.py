import os
import inspect
from glob import glob
from shutil import copyfile

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)


def copy_files(path: str, files_to_copy: list, sys_path_delimiter: str):
    files = []
    for file in files_to_copy:
        files += glob(os.path.join(parentdir, *file))

    path_ = os.path.join(path, 'py_Files')
    if not os.path.exists(path_):
        os.mkdir(path_)

    for file in files:
        if os.path.exists(file):
            copyfile(file, os.path.join(path_, file.split(sys_path_delimiter)[-1]))
