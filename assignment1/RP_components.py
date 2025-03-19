# Contain components for ResolutionProp

def is_neg(l: str) -> bool:
    return l.startswith('~')

def is_diff(l: str, r: str) -> bool:
    return is_neg(l) ^ is_neg(r)

def to_pos(l: str) -> str:
    return l[1:] if is_neg(l) else l

def to_neg(l: str) -> str:
    return l if is_neg(l) else '~' + l
