from typing import Optional, Tuple


def split_str_on(s: str, c: str):
    special = "\"`'"

    assert len(c) == 1
    assert c not in special

    looking_for = ""
    splits = []
    prev_ind = 0
    for ind, cx in enumerate(s):
        if looking_for:
            if looking_for == cx:
                looking_for = ""
            else:
                continue
        elif cx in special:
            looking_for = cx
        elif cx == c:
            splits.append(s[prev_ind:ind])
            prev_ind = ind + 1
    splits.append(s[prev_ind:])
    return splits


class Section:
    def __init__(self, header: Optional[str] = None):
        self.header = header
        self.lines = []
        self.body = None
        self.sentences = None

    def push_line(self, line: str):
        self.lines.append(line)

    def __len__(self):
        return len(self.lines)

    def consolidate(self):
        self.body = " ".join(self.lines)
        self.sentences = [s for s in split_str_on(self.body, ".") if s]


class Docs:
    def __init__(self):
        self._sections: [Tuple[Optional[str], Section]] = []
        self._consolidated = False

    def consolidate(self):
        if not self._consolidated:
            for section in self._sections:
                section.consolidate()
        self._consolidated = True

    def push_line(self, line: str):
        line_strip = line.strip("\" ")
        if line_strip.startswith("#"):
            self._sections.append(Section(line))
        else:
            if not self._sections:
                self._sections.append(Section(None))
            self._sections[-1].push_line(line_strip)
        self._consolidated = False

    def sections(self) -> [Section]:
        self.consolidate()
        return self._sections

    def __str__(self):
        lines = []
        for section in self.sections():
            if section.header:
                lines.append(section.header)
            lines += section.lines
        return "\n".join(f"/// {line}" for line in lines)


if __name__ == '__main__':
    print(split_str_on("hello `there`. general `kenobi`. Hello. ...", "."))
    print("hello `there`. general `kenobi`. Hello. ...".split("."))
    print("hello `the.re`. general `kenobi`. Hello. ...".split("."))
    print(split_str_on("hello `the.r\"e\"`. general `kenobi`. Hello. ...", "."))