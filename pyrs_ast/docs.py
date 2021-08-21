from typing import Optional, Tuple
import re

CODE_REGEX = re.compile(r"\.(?=(?:[^\`']*[\`'][^\`']*[\`'])*[^\`']*$)")


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
        self.sentences = [s for s in CODE_REGEX.split(self.body) if s]


class Docs:
    def __init__(self):
        self._sections: [Tuple[Optional[str], Section]] = []
        self._consolidated = False

    def _consolidate(self):
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
        self._consolidate()
        return self._sections
