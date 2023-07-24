import os
import inspect
import platform

import pytest


@pytest.fixture
def main_dir() -> str:
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


@pytest.fixture
def sys_slash() -> str:
    if platform.system() == 'Windows':
        return "\\"
    else:
        return "/"
