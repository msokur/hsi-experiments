import abc

from struct import unpack


s_SIZE = 1
d_SIZE = 8
I_SIZE = 4
b_SIZE = 1
ENCODING = "iso-8859-1"


class MK_base:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = open(self.file_path, "rb").read()

    def load(self):
        index, left, top, colo, radius, text, spec, last_byte = self.load_marker()
        in_class, last_byte = self.marker_in_class(start_byte=last_byte)
        names = self.load_string(start_byte=last_byte)

        return names, left, top, radius, index

    def load_string(self, start_byte: int):
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
                names.append(None)

        return names

    @abc.abstractmethod
    def load_marker(self):
        pass

    @abc.abstractmethod
    def marker_in_class(self, start_byte: int):
        pass


class MK2(MK_base):
    def load_marker(self):
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

    def marker_in_class(self, start_byte: int):
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
    path = r"E:\ICCAS\tools\CreateMarkerFile"
    file = "test_marker3.mk2"

    marker_ = MK2(os.path.join(path, file))
    datas = marker_.load()
    v = 1
