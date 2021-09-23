from doc_parser.nlp_query import is_one_of

PREDICATE = "#FF8B3D"
A1 = "#ADD8E6"
A2 = "#00AA00"

ENTITY_COLORS = {
    "predicate": PREDICATE,
    "A1": A1,
    "A2": A2
}

NOUN = '#00AA00'
VERB = '#FF8B3D'
PREP = '#E3242B'
MODIFIER = '#902010'
COORD = '#5151b2'
FADE = '#b0b0b0'

TREE_COLORS = {
    "BOOL_EXPR": PREP,
    "QASSERT": PREP,
    "HASSERT": PREP,
    "COND": PREP,
    "FOR": FADE
}


def tag_color(pos: str) -> str:
    """Returns a color code for the part of speech."""
    if pos is None:
        return "#000000"

    if is_one_of(pos, {"VB", "RET", "MD", "VP"}):
        return VERB
    if is_one_of(pos, {"RB"}):
        return "#ff9a57"
    if is_one_of(pos, {"PRP", "NN", "NP", "CODE", "OBJ", "LIT"}):
        return NOUN
    if is_one_of(pos, {"JJ"}):
        return "#00dd00"
    if is_one_of(pos, {"IN", "TO", "PP", "IS", "IF"}):
        return PREP
    if is_one_of(pos, {"DT", "CC"}):
        return FADE
    if is_one_of(pos, {"CC"}):
        return COORD

    return TREE_COLORS.get(pos, "#000000")
