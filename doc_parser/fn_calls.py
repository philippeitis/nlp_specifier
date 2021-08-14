from nltk import Tree


class Rule:
    def __init__(self, word: str):
        body = word[1:-1]
        self.opt = False
        if ":" not in body:
            self.opt = body.startswith("?")
            self._symbol = body.lstrip("?")
            self.ident = None
        else:
            self.ident, self._symbol = body.split(":")

    def __str__(self):
        return f"?{{{self.ident}:{self._symbol}}}"

    def symbol(self):
        return self._symbol

    def is_optional(self):
        return self.opt


class Literal:
    def __init__(self, s: str):
       self.word = s

    def symbol(self):
        return f"\"{self.word}\""

    def is_optional(self):
        return False


class InvokeToken:
    def __init__(self, word: str, is_symbol: bool, **kwargs):
        if is_symbol:
            self.word = Rule(word)
        else:
            self.word = Literal(word)
        self.kwargs = kwargs

    def __str__(self):
        return f"{self.word}"

    def symbol(self):
        return self.word.symbol()

    def is_optional(self):
        return self.word.is_optional()


class InvocationFactory:
    def __init__(self, call_back):
        self.invocations = {}
        self.productions = {}
        self.initializers = {}
        self.call_back = call_back

    def add_invocation(self, fn: str, invocation: str):
        invocation = Invocation(fn, invocation)
        if fn in self.invocations:
            num = len(self.invocations[fn])
        else:
            num = 0
            self.invocations[fn] = []
        for constructor, grammar in invocation.constructors(self.call_back):
            name = f"FN_{fn.upper()}_{num}"
            self.invocations[fn].append(name)
            self.productions[name] = f"{name} -> {' '.join(x.symbol() for x in grammar)}"
            num += 1
            self.initializers[name] = constructor

    def grammar(self):
        rules = []
        productions = []
        for fn, invocations in self.invocations.items():
            rules.extend(invocations)
            productions.extend((self.productions[invoke] for invoke in invocations))

        rule_line = f"FNCALL -> {' | '.join(rules)}"
        productions = "\n".join(productions)
        return f"{rule_line}\n{productions}"

    def __call__(self, tree: Tree):
        return self.initializers[tree[0].label()](tree[0])


class Invocation:
    def __init__(self, fn: str, sentence: str):
        self.fn = fn
        parts = []
        a = 0
        while a < len(sentence) and sentence[a].isspace():
            a += 1

        while a < len(sentence):
            is_symbol = False
            if sentence[a] == "{":
                is_symbol = True
                ind = sentence[a + 1:].find("}")
                if ind != -1:
                    ind += 2
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
                t = InvokeToken(sentence[a:], is_symbol, whitespace_after=False, start_position=a)
                parts.append(t)
                break
            else:
                t = InvokeToken(sentence[a: a + ind], is_symbol, whitespace_after=True, start_position=a)
                parts.append(t)

            a = a + ind
            while a < len(sentence) and sentence[a] in {",", " "}:
                # if sentence[a] == ",":
                #     t = Token(sentence[a], num, whitespace_after=False, start_position=a)
                #     parts.append(t)
                #     num += 1

                a += 1

        self.parts = parts
        self.optional_tokens = [x for x in self.parts if x.is_optional()]

        self.inputs = {}
        # TODO: Need to add tests against actual Fn class
        for part in parts:
            if isinstance(part.word, Rule) and not part.is_optional():
                self.inputs[part.word.ident] = part.symbol()

    def __str__(self):
        return " ".join(str(x) for x in self.parts)

    def grammar_variants(self):
        if self.optional_tokens:
            for i in range(0, 2**len(self.optional_tokens)):
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

    def constructors(self, call_back):
        for variant in self.grammar_variants():
            yield call_back(self.fn, variant), variant
