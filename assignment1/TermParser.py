# Include the definition of Term, Variable, Constant, Function and the parser for term
# TermParser.py

# set of names of variables 
VARIABLES = {'u', 'v', 'w', 'x', 'y', 'z', 'uu', 'vv', 'ww', 'xx', 'yy', 'zz'}

# Term class and its subclasses
class Term:
    def __init__(self):
        pass

    def __eq__(self, other):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

class Variable(Term):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name

    def __repr__(self):
        return self.name

class Constant(Term):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Constant) and self.name == other.name

    def __repr__(self):
        return self.name

# All functions and predicates are represented as Function
class Function(Term):
    def __init__(self, name, args):
        self.name = name
        self.args = args  # list of Terms

    def __eq__(self, other):
        return (isinstance(other, Function) and 
                self.name == other.name and 
                len(self.args) == len(other.args) and
                all(a == b for a, b in zip(self.args, other.args)))

    def __repr__(self):
        return f"{self.name}(" + ",".join(repr(arg) for arg in self.args) + ")"

# Paser: parse string to term
def parse_term(s):
    # remove spaces
    s = s.replace(" ", "")
    index = 0

    def parse():
        nonlocal index
        # read identifier consisting of characters
        start = index
        while index < len(s) and s[index].isalpha():
            index += 1
        name = s[start:index]
        # if the next character is '(', it is considered a function/predicate
        if index < len(s) and s[index] == '(':
            index += 1  # skip '('
            args = []
            while True:
                args.append(parse())
                if index < len(s) and s[index] == ',':
                    index += 1  # skip ','
                elif index < len(s) and s[index] == ')':
                    index += 1  # skip ')'
                    break
                else:
                    break
            return Function(name, args)
        # the identifier is considered a variable if in VARIABLES, otherwise treated as a constant
        return Variable(name) if name in VARIABLES else Constant(name)
    
    return parse()

# Apply a substitution to a term
def apply_subst(term, subst):
    """Apply a substitution to a term.
    Args:
        term: Term
        subst: dict, variable name -> Term
    Returns:
        Term
    """
    if isinstance(term, Variable):
        if term.name in subst:
            # recursively apply the substitution
            return apply_subst(subst[term.name], subst)
        else:
            return term
    elif isinstance(term, Constant):
        return term
    elif isinstance(term, Function):
        new_args = [apply_subst(arg, subst) for arg in term.args]
        return Function(term.name, new_args)
    else:
        return term