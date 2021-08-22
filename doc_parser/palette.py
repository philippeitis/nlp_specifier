from nlp_query import is_one_of

PREDICATE = "#FF8B3D"
A1 = "#ADD8E6"
A2 = "#00AA00"

ENTITY_COLORS = {
    "predicate": PREDICATE,
    "A1": A1,
    "A2": A2
}

NOUN = '#700070'
VERB = '#ff9a57'
PREP = '#C35617'
MODIFIER = '#902010'
COORD = '#404090'
FADE = '#b0b0b0'


def tag_color(pos: str) -> str:
    """Returns a color code for the part of speech."""
    if pos is None:
        return "#000000"

    if is_one_of(pos, {"VB", "RET", "MD", "VP"}):
        return VERB
    if is_one_of(pos, {"PRP", "NN", "NP", "CODE", "OBJ", "LIT"}):
        return NOUN
    if is_one_of(pos, {"IN", "TO", "PP"}):
        return PREP
    if is_one_of(pos, {"DT"}):
        return FADE
    if is_one_of(pos, {"RB", "JJ", "ADVP", "ADJP"}):
        return MODIFIER
    if is_one_of(pos, {"CC"}):
        return COORD

    return "#000000"
