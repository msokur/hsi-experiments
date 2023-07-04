import pytest
import os
import numpy as np
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)

from data_utils.marker import MK2

MK2_LOAD_MARKER_DATA = [("test_mask.mk2",
                         [0., 1., 2., 3., float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                          float("nan")],
                         [1., 5., 1., 5., 0., 0., 0., 0., 0., 0.],
                         [1., 5., 5., 1., 0., 0., 0., 0., 0., 0.],
                         [123, 156, 212, 3423, 0, 0, 0, 0, 0, 0],
                         [1, 2, 0, 1, 0, 0, 0, 0, 0, 0],
                         ["--", "--", "--", "--", "--", "--", "--", "--", "--", "--"],
                         [None, None, None, None, None, None, None, None, None, None],
                         394),
                        ("test_marker.mk2",
                         [0., 1., 2.],
                         [2., 33., 1.],
                         [12., 11., 1.],
                         [1213, 161, 1],
                         [20, 2, 1],
                         ["one", "two", "three"],
                         [[0, 1, 2, 3], None, [4, 5]],
                         174),
                        ("test_marker2.mk2",
                         [0.],
                         [0.],
                         [0.],
                         [0],
                         [0],
                         ["äöüß?/\\"],
                         [None],
                         48)]


@pytest.mark.parametrize("file,r_idx,r_left,r_top,r_color,r_radius,r_text,r_spec,r_end_byte", MK2_LOAD_MARKER_DATA)
def test_mk2_load_maker(file, r_idx, r_left, r_top, r_color, r_radius, r_text, r_spec, r_end_byte):
    loader = MK2(file_path=os.path.join(parent_dir, "data_utils", "data_loaders", "test_data", "dat_file", file))
    idx, left, top, color, radius, text, spec, end_byte = loader.load_marker()

    assert [x == y if not np.isnan(x) else True for x, y in zip(idx, r_idx)]
    assert left == r_left
    assert top == r_top
    assert color == r_color
    assert radius == r_radius
    assert text == r_text
    assert spec == r_spec
    assert end_byte == r_end_byte


MK2_MARKER_IN_CLASS_DATA = [("test_mask.mk2", 394, [], 398),
                            ("test_marker.mk2", 174, [13, 1, 2], 190),
                            ("test_marker2.mk2", 48, [], 52)]


@pytest.mark.parametrize("file,start_byte,result,r_end_byte", MK2_MARKER_IN_CLASS_DATA)
def test_mk2_marker_in_class(file, start_byte, result, r_end_byte):
    loader = MK2(file_path=os.path.join(parent_dir, "data_utils", "data_loaders", "test_data", "dat_file", file))
    in_class, end_byte = loader.marker_in_class(start_byte=start_byte)

    assert in_class == result
    assert end_byte == r_end_byte


MK2_LOAD_STRING_DATA = [("test_mask.mk2", 398,
                         ["Class0", "Class1", "Class0", "Class2", "", "", "", "", "", ""]),
                        ("test_marker.mk2", 190,
                         ["class one", "class two", "class three", "", "", "", "", "", "", ""]),
                        ("test_marker2.mk2", 52,
                         ["ä", "", "", "", "", "", "", "",  "",  ""])]


@pytest.mark.parametrize("file,start_byte,result", MK2_LOAD_STRING_DATA)
def test_mk2_load_string(file, start_byte, result):
    loader = MK2(file_path=os.path.join(parent_dir, "data_utils", "data_loaders", "test_data", "dat_file", file))
    names = loader.load_string(start_byte=start_byte)

    assert names == result
