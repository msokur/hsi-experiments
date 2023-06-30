import pytest
import os
import numpy as np

from data_utils.marker import MK2


@pytest.fixture
def mk2_marker():
    return MK2(os.path.join("data_loaders", "test_data", "dat_file", "test_mask.mk2"))


def test_mk2_load_maker(mk2_marker):
    idx, left, top, radius, begin_marker, start_byte = mk2_marker.load_marker()

    assert (idx[:4] == np.array([0., 1., 2., 3.])).all() and np.isnan(idx[4:]).all()
    assert (left == [1., 5., 1., 5., 0., 0., 0., 0., 0., 0.]).all()
    assert (top == [1., 5., 5., 1., 0., 0., 0., 0., 0., 0.]).all()
    assert (radius == [1, 2, 0, 1, 0, 0, 0, 0, 0, 0]).all()
    assert begin_marker == 0
    assert start_byte == 400


def test_mk2_load_string(mk2_marker):
    assert mk2_marker.load_string(400, 0) == ["Class0", "Class1", "Class0", "Class2"]
