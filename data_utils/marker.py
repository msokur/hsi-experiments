import abc

import numpy as np
from struct import unpack
import math


markers = 10
char_obj = ["?", "(", ",", ")"]


class MK_base:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = open(self.file_path, "rb").read()

    def load(self):
        index, left, top, radius, begin_marker, start_byte = self.load_marker()
        names = self.load_string(start_byte, begin_marker)
        if len(names) > 11:
            names = self.load_string(8000, 0)

        return names, left, top, radius, index

    def load_string(self, start_byte: int, begin_marker: int):
        marker_name1 = []
        for idx in range(start_byte, len(self.data)):
            exceptV = 0
            string_marker = chr(self.data[idx])
            if begin_marker == 1 and string_marker == " ":
                exceptV = 1

            if (str.isalpha(string_marker) or str.isnumeric(string_marker) or string_marker in char_obj) \
                    and string_marker != "ÿ" or exceptV == 1:

                if self.data[idx - 2] == 0 and self.data[idx - 3] == 0 and self.data[idx - 4] == 0 or begin_marker == 1:
                    if begin_marker != 1:
                        marker_name1.append("/")

                    marker_name1.append(string_marker)

                    if (idx + 1 and idx + 2) < len(self.data):
                        if self.data[idx + 1] == 0 and self.data[idx + 2] == 0:
                            begin_marker = 0
                        else:
                            begin_marker = 1

        maker_name2 = "".join(str(x) for x in marker_name1)
        return maker_name2.split("/")[1:]

    @abc.abstractmethod
    def load_marker(self):
        pass


class MK2(MK_base):
    def load_marker(self):
        index = np.zeros(markers)
        left = np.zeros(markers)
        top = np.zeros(markers)
        radius = np.zeros(markers)

        begin_marker = 0
        start_byte = 400

        data_noinfo = 4
        mark = 0
        for idx in range(markers):
            datastruct = ">dddIb"
            if idx == 0:
                s = unpack(datastruct, self.data[data_noinfo:(29 + data_noinfo)])
                mark = 1
            else:
                s = unpack(datastruct, self.data[idx * (30 + 9) + data_noinfo:29 + idx * (30 + 9) + data_noinfo])
                if (not math.isnan(s[0]) or s[1] < 690 and s[0] < 0 or s[0] == 1) and (0 < s[4] < 600) and idx == 1:
                    mark = 2
                if (math.isnan(s[0]) or s[0] > 600 or s[0] < 0) and s[4] != 20 and idx > 0 and mark == 1 and mark != 2 and len(self.data) > (29 + (idx * ((49 * 16) + 25 + 30)) + data_noinfo):
                    mark = 1
                    s = unpack(datastruct, self.data[(idx * (49 * 16 + 25 + 30)) + data_noinfo:(29 + (idx * (49 * 16 + 25 + 30)) + data_noinfo)])

            index[idx] = s[0]
            left[idx] = s[1]
            top[idx] = s[2]
            radius[idx] = s[4]

        return index, left, top, radius, begin_marker, start_byte


if __name__ == "__main__":
    import os
    path = r"E:\ICCAS\tools\CreateMarkerFile"
    file = "test_mask.mk2"

    marker = MK2(os.path.join(path, file))
    datas = marker.load()
    v = 1
