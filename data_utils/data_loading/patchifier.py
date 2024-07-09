import numpy as np

from skimage.util.shape import view_as_windows


def __pad_width(patch_size: tuple, with_axis_2: bool) -> list:
    pad = [int((s - 1) / 2) for s in patch_size]
    pad_width = [[pad[idx], pad[idx]] if s % 2 == 1 else [pad[idx], pad[idx] + 1] for idx, s in enumerate(patch_size)]
    if with_axis_2:
        pad_width.append([0, 0])

    return pad_width


def patching_as_view(cube: np.ndarray, patch_size: tuple | list) -> np.ndarray:
    pad_cube = np.pad(array=cube,
                      pad_width=__pad_width(patch_size=patch_size,
                                            with_axis_2=len(cube.shape) > 2))

    window_shape = (patch_size[0], patch_size[1], cube.shape[2] if len(cube.shape) > 2 else 1)
    patches = view_as_windows(arr_in=pad_cube,
                              window_shape=window_shape,
                              step=1).squeeze()

    return patches
