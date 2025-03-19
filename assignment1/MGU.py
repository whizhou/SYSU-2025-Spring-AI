# 人工智能 lab1 最一般合一算法
# 23336020 周子健

import TermParser
from TermParser import parse_term, apply_subst, Variable, Function

def occurs(var, term, subst):
    """Check if var occurs in term after applying subst.
    Args:
        var: Variable
        term: Term
        subst: dict, variable name -> Term
    Returns:
        bool
    """
    term = apply_subst(term, subst)
    if isinstance(term, Variable):
        return term.name == var.name
    elif isinstance(term, Function):
        return any(occurs(var, arg, subst) for arg in term.args)
    return False


def unify(t1, t2):
    """Most General Unifier
    Args:
        t1: Term
        t2: Term
    Returns:
        dict, variable name -> Term
        or None if unification fails
    """
    # equations: list of pairs of terms
    equations = [(t1, t2)]
    # sigma: dict, variable name -> Term
    sigma = {}

    while equations:
        s, t = equations.pop(0)
        # 对 s 和 t 应用当前的替换 sigma
        s = apply_subst(s, sigma)
        t = apply_subst(t, sigma)
        
        if s == t:
            continue
        elif isinstance(s, Variable):
            if occurs(s, t, sigma):
                return None
            # place s with t
            sigma[s.name] = t
            equations = [(apply_subst(l, {s.name: t}), apply_subst(r, {s.name: t})) for l, r in equations]
            # update sigma
            sigma = {var: apply_subst(term, {s.name: t}) for var, term in sigma.items()}
        elif isinstance(t, Variable):
            # symmetric case
            if occurs(t, s, sigma):
                return None
            sigma[t.name] = s
            equations = [(apply_subst(l, {t.name: s}), apply_subst(r, {t.name: s})) for l, r in equations]
            sigma = {var: apply_subst(term, {t.name: s}) for var, term in sigma.items()}
        elif isinstance(s, Function) and isinstance(t, Function):
            if s.name != t.name or len(s.args) != len(t.args):
                return None
            else:
                # add sub-equations
                equations = list(zip(s.args, t.args)) + equations
        else:
            # not match
            return None
    return sigma


def MGU(str1, str2):
    """Main function for Most General Unifier.
    Args:
        str1: str, term 1
        str2: str, term 2
    Returns:
        dict, variable name -> Term
        str, unified term
    """
    term1 = parse_term(str1)
    term2 = parse_term(str2)
    sigma = unify(term1, term2)
    if sigma is None:
        return None
    term1 = apply_subst(term1, sigma)
    term2 = apply_subst(term2, sigma)
    assert term1.__repr__() == term2.__repr__()
    return sigma, term1.__repr__()


# test examples
if __name__ == '__main__':
    input1 = ['P(xx,a)', 'P(a,xx,f(g(yy)))', 'P(a,x,h(g(z)))']
    input2 = ['P(b,yy)', 'P(zz,f(zz),f(uu))', 'P(z,h(y),h(y))']
    test_id = 1
    mgu_result = MGU(input1[test_id], input2[test_id])
    if mgu_result is None:
        print("Failed to unify.")
    else:
        # print("Unification success.")
        sigma, term = mgu_result
        print(sigma)
        # for var, term in sigma.items():
            # print(f"  {var} -> {term}")
