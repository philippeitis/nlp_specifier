import re

UNSIGNED_RE = re.compile("([\d+_]+)(_?u(8|16|32|64|128|size))?")
INTEGER_RE = re.compile("(-)?([\d_]+)(_?i(8|16|32|64|128|size))?")
FLOAT_RE = re.compile("(-)?([\d_]*)\.?([\d_]*)(e(-?)(\d*_))?(_?(f32|f64))?")
NUM_REGEX = re.compile(
    ""
)


class UInteger:
    def __init__(self, matches):
        self.num = matches[0]
        self.width = matches[1]


class Integer:
    def __init__(self, matches):
        self.is_pos = bool(matches[0])
        self.num = matches[1]
        self.width = matches[2]


class Float:
    def __init__(self, matches):
        self.sign = matches[0] or ""
        self.big = matches[1]
        self.small = matches[2]
        self.e_sign = matches[4] or ""
        self.e_num = matches[5]

        self.width = matches[7]

    def __str__(self):
        e_num = self.e_num or ""
        e = f"e{self.e_sign}{e_num}" if self.e_num else ""
        big = self.big or ""
        small = self.small or ""
        return f"{self.sign}{big}.{small}{e}_{self.width}"

if __name__ == '__main__':
    print(INTEGER_RE.match("-7_000_000_i8").groups())
    print(INTEGER_RE.match("7_000_000_usize").groups())
    print(UNSIGNED_RE.match("7_000_000_usize").groups())

    print(Float(FLOAT_RE.match("-32.3e-10_f32").groups()))
