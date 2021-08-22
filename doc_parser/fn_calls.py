import logging
from collections import defaultdict

from nltk import Tree

from pyrs_ast.lib import Fn

LOGGER = logging.getLogger(__name__)


class Rule:
    def __init__(self, word: str, is_optional: bool = False):
        body = word[1:-1]
        self.optional = is_optional
        if ":" not in body:
            self.optional = body.startswith("?")
            self._symbol = body.lstrip("?")
            self.ident = None
        else:
            self.ident, self._symbol = body.split(":")

    def __str__(self):
        if self.ident:
            body = f"{{{self.ident}:{self.symbol()}}}"
        else:
            body = f"{{{self._symbol}}}"

        if self.optional:
            return f"?{body}"
        return body

    def symbol(self) -> str:
        return self._symbol

    def is_optional(self) -> bool:
        return self.optional


class Literal:
    def __init__(self, s: str, is_optional: bool = False):
        self.word = s
        self.optional = is_optional

    def symbol(self) -> str:
        return f"\"{self.word}\""

    def is_optional(self) -> bool:
        return self.optional

    def __str__(self):
        return self.symbol()


class InvokeToken:
    def __init__(self, word: str, is_symbol: bool, is_optional: bool = False):
        if is_symbol:
            self.word = Rule(word)
        else:
            self.word = Literal(word, is_optional)

    def __str__(self):
        return f"{self.word}"

    def symbol(self):
        return self.word.symbol()

    def is_optional(self):
        return self.word.is_optional()


class InvocationFactory:
    def __init__(self, callback):
        self.invocations = defaultdict(list)
        self.productions = {}
        self.initializers = {}
        self.callback = callback

    def add_invocation(self, fn: Fn, invocation: "Invocation"):
        invocations = self.invocations[fn.ident]
        num = len(invocations)
        for constructor, grammar in invocation.constructors(self.callback):
            name = f"FN_{fn.ident}_{num}"
            invocations.append(name)
            self.productions[name] = f"{name} -> {' '.join(x.symbol() for x in grammar)}"
            self.initializers[name] = constructor
            num += 1

    def add_fuzzy_invocation(self, fn: Fn, labels, words):
        invoke_tokens = []
        token_it = enumerate(zip(labels, words))
        idents = {ty.ident for ty in fn.inputs}
        LOGGER.info(f"Adding fuzzy invocation for fn {fn.ident}")

        for i, (label, word) in token_it:
            if label in {"RET", "COMMA", "DT"}:
                continue

            if label in {"CODE"}:
                word_strip = word.strip("`")

                if word.strip("`") in idents:
                    sym = f"{{{word_strip}:OBJ}}"
                else:
                    sym = "{OBJ}"
                invoke_tokens.append(InvokeToken(sym, is_symbol=True, is_optional=False))
            elif label in {"LIT"}:
                invoke_tokens.append(InvokeToken(f"{{{word}:OBJ}}", is_symbol=True, is_optional=False))
            else:
                invoke_tokens.append(InvokeToken(word, is_symbol=False, is_optional=False))
            break

        for i, (label, word) in token_it:
            if label in {"RET", "COMMA", "DOT"}:
                continue

            if label in {"CODE"}:
                word_strip = word.strip("`")

                if word.strip("`") in idents:
                    sym = f"{{{word_strip}:OBJ}}"
                else:
                    sym = "{OBJ}"
                invoke_tokens.append(InvokeToken(sym, is_symbol=True, is_optional=False))
            elif label in {"LIT"}:
                invoke_tokens.append(InvokeToken(f"{{{word}:OBJ}}", is_symbol=True, is_optional=False))
            else:
                invoke_tokens.append(InvokeToken(word, is_symbol=False, is_optional=False))

        invocation = Invocation(fn, invoke_tokens)
        self.add_invocation(fn, invocation)

    def grammar(self):
        rules = []
        productions = []
        for fn, invocations in self.invocations.items():
            rules.extend(invocations)
            productions.extend((self.productions[invoke] for invoke in invocations))

        rule_line = f"FNCALL -> {' | '.join(rules)}"
        productions = "\n".join(productions)
        return f"{rule_line}\n{productions}"

    def __call__(self, tree: Tree, *args):
        return self.initializers[tree[0].label()](tree[0], self)


class Invocation:
    def __init__(self, fn: Fn, parts):
        self.fn = fn
        self.parts = parts
        self.optional_tokens = [x for x in self.parts if x.is_optional()]

    @classmethod
    def from_sentence(cls, fn: Fn, sentence: str):
        parts = []
        a = 0
        while a < len(sentence) and sentence[a].isspace():
            a += 1

        while a < len(sentence):
            is_symbol = False
            is_code = False
            if sentence[a] in ("{", "`"):
                LOGGER.info(f"Found symbol")
                opposite = {"{": "}", "`": "`"}[sentence[a]]
                is_symbol = True
                ind = sentence[a + 1:].find(opposite)
                if ind != -1:
                    ind += 2
                if sentence[a] == "`":
                    is_code = True

            else:
                sind = sentence[a:].find(" ")
                cind = sentence[a:].find(",")
                if cind == -1:
                    ind = sind
                elif sind == -1:
                    ind = cind
                else:
                    ind = min(sind, cind)
            if ind == -1:
                if is_code:
                    t = InvokeToken("{OBJ}", True)
                else:
                    t = InvokeToken(sentence[a:], is_symbol)
                parts.append(t)
                break
            else:
                if is_code:
                    t = InvokeToken("{OBJ}", True)
                else:
                    t = InvokeToken(sentence[a: a + ind], is_symbol)

                parts.append(t)

            a += ind
            while a < len(sentence) and sentence[a] in {",", " "}:
                a += 1

        return cls(fn, parts)

    def __str__(self):
        return " ".join(str(x) for x in self.parts)

    def grammar_variants(self):
        if self.optional_tokens:
            for i in range(0, 2 ** len(self.optional_tokens)):
                variant = []
                for token in self.parts:
                    if token.is_optional():
                        if i & 1 == 0:
                            i >>= 1
                            continue
                    variant.append(token)
                yield variant
        else:
            yield self.parts

    def as_grammar(self, name: str):
        return f"{name} -> {' | '.join(' '.join(x.symbol() for x in variant) for variant in self.grammar_variants())}"

    def constructors(self, callback):
        for variant in self.grammar_variants():
            yield callback(self.fn, variant), variant
