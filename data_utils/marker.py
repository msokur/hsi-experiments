import abc

from struct import unpack
from typing import Tuple, List, Any, Optional

s_SIZE = 1
d_SIZE = 8
I_SIZE = 4
b_SIZE = 1
ENCODING = "iso-8859-1"


class MK_base:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = open(self.file_path, "rb").read()

    def load(self) -> Tuple[list[str], list[float], list[float], list[int], list[float]]:
        """
        Loads the marker and returns 5 values for the annotation.

        :return:  Tuple with a list with names, a list with the coordinates to the left, a list with the coordinates
        to the top, a list with the radius for every marker and a list with the index from every marker.
        """
        index, left, top, colo, radius, text, spec, last_byte = self.load_marker()
        in_class, last_byte = self.marker_in_class(start_byte=last_byte)
        names = self.load_string(start_byte=last_byte)

        return names, left, top, radius, index

    def load_string(self, start_byte: int) -> List[str]:
        """
        Loads the strings from the last part of the file.

        :param start_byte: The byte at which the read operation should begin.

        :return: A list with the strings.
        """
        name_length, names = [], []

        while start_byte < len(self.data):
            fmt = ">I"
            end_byte = start_byte + I_SIZE
            name_length.append(unpack(fmt, self.data[start_byte:end_byte])[0])
            start_byte = end_byte

            if name_length[-1] > 0:
                end_byte += s_SIZE * name_length[-1]
                names.append(self.data[start_byte:end_byte].decode(encoding=ENCODING))
                start_byte = end_byte
            else:
                names.append("")

        return names

    @abc.abstractmethod
    def load_marker(self) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any], list[str],
                                   list[Optional[list[Any]]], int]:
        pass

    @abc.abstractmethod
    def marker_in_class(self, start_byte: int) -> Tuple[list[Optional[int]], int]:
        pass


class MK2(MK_base):
    """ Read MK2 marker files
    From byte 0 - 4 is the integer with number of markers. For every marker there is a fix section with values.
    Every marker has the following structure:
    1. 8 byte float with 'Index' value.
    2. 8 byte float with 'Left' value.
    3. 8 byte float with 'Top' value.
    4. 4 byte integer with 'Color' value.
    5. 1 byte integer with 'Radius' value.
    6. 4 byte integer with the number of letters in the text section.
    7. 1 byte char for every letter in text section. (Value from 6. * 1 byte)
    8. 4 byte integer with the number of values in MeanSpect
    9. 8 byte float for every value in MeanSpect. (Value from 8. * 8 byte)

    After the marker section, there are 4 bytes with an integer containing the number of markers that are used and
    there classification number.
    1. 4 byte integer with classification number.

    In the last part there are 10 strings. Every string has following structure:
    1. 4 byte integer with the length of the string.
    2. 1 byte char for every letter in the string. (Value from 1. * 1 byte)
    """

    def load_marker(self) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any], list[str],
                                   list[Optional[list[Any]]], int]:
        """
        Loads the markers with their index, left and top position, color, radius, text and mean spectrum.

        :return: Tuple with the all markers index, left and top positions, color, radius, text, mean spectrum and the
        last byte that was read.
        """
        index, left, top, color, radius, text_length, text, spec_depth, spec = [], [], [], [], [], [], [], [], []

        start_byte = 0
        end_byte = I_SIZE
        fmt = ">I"
        marker = unpack(fmt, self.data[start_byte:end_byte])[0]
        start_byte = end_byte

        fix_step = d_SIZE + d_SIZE + d_SIZE + I_SIZE + b_SIZE + I_SIZE

        for idx in range(marker):
            end_byte = start_byte + fix_step
            fmt = ">dddIbI"
            fix_part = unpack(fmt, self.data[start_byte:end_byte])
            start_byte = end_byte
            index.append(fix_part[0])
            left.append(fix_part[1])
            top.append(fix_part[2])
            color.append(fix_part[3])
            radius.append(fix_part[4])
            text_length.append(fix_part[5])

            if text_length[-1] > 0:
                end_byte += s_SIZE * text_length[-1]
                text.append(self.data[start_byte:end_byte].decode(encoding=ENCODING))
                start_byte = end_byte
            else:
                text.append("")

            end_byte += I_SIZE
            fmt = ">I"
            spec_depth.append(unpack(fmt, self.data[start_byte:end_byte])[0])
            start_byte = end_byte

            if spec_depth[-1] > 0:
                end_byte += d_SIZE * spec_depth[-1]
                fmt = ">" + ("d" * spec_depth[-1])
                spec_part = unpack(fmt, self.data[start_byte:end_byte])
                spec.append([val for val in spec_part])
                start_byte = end_byte
            else:
                spec.append(None)

        return index, left, top, color, radius, text, spec, end_byte

    def marker_in_class(self, start_byte: int) -> Tuple[list[Optional[int]], int]:
        """
        Loads the classification number for every marker.

        :params start_byte: The byte at which the read operation should begin.

        :return: A list with the classification numbers and the last byte that was read.
        """
        in_class = []

        fmt = ">I"
        end_byte = start_byte + I_SIZE
        classes = unpack(fmt, self.data[start_byte:end_byte])[0]
        start_byte = end_byte

        for idx in range(classes):
            fmt = ">I"
            end_byte += I_SIZE
            in_class.append(unpack(fmt, self.data[start_byte:end_byte])[0])
            start_byte = end_byte

        return in_class, end_byte


if __name__ == "__main__":
    import os

    #path = r"E:\ICCAS\tools\CreateMarkerFile"
    #file = "test_marker3.mk2"
    path = r"D:\ICCAS\Gastric\General\annotation\mk_files"
    file = "Laura_Daten11.mk2"

    marker_ = MK2(os.path.join(path, file))
    datas = marker_.load()
    v = 1
