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
    else:
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
                # 循环引用，合一失败
                return None
            else:
                # 记录替换 s -> t
                sigma[s.name] = t
                # 将该替换应用到剩余的方程中
                new_equations = []
                for l, r in equations:
                    new_l = apply_subst(l, {s.name: t})
                    new_r = apply_subst(r, {s.name: t})
                    new_equations.append((new_l, new_r))
                equations = new_equations
                # 同时更新已有替换
                for var in sigma:
                    sigma[var] = apply_subst(sigma[var], {s.name: t})
        elif isinstance(t, Variable):
            # 对称处理
            if occurs(t, s, sigma):
                return None
            else:
                sigma[t.name] = s
                new_equations = []
                for l, r in equations:
                    new_l = apply_subst(l, {t.name: s})
                    new_r = apply_subst(r, {t.name: s})
                    new_equations.append((new_l, new_r))
                equations = new_equations
                for var in sigma:
                    sigma[var] = apply_subst(sigma[var], {t.name: s})
        elif isinstance(s, Function) and isinstance(t, Function):
            if s.name != t.name or len(s.args) != len(t.args):
                return None
            else:
                # 将 s 与 t 分解为各对应的参数方程
                equations = list(zip(s.args, t.args)) + equations
        else:
            # 不匹配的情况
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
    result = unify(term1, term2)
    if result is None:
        return None
    term1 = apply_subst(term1, result)
    term2 = apply_subst(term2, result)
    assert term1.__repr__() == term2.__repr__()
    return result, term1.__repr__()


# test examples
if __name__ == '__main__':
    input1 = ['P(xx, a)', 'P(a,xx, f(g(yy)))', 'P(a,x,h(g(z)))', 'first']
    input2 = ['P(b, yy)', 'P(zz, f(zz),f(uu))', 'P(z,h(y),h(y))', '~first']
    input_dim = 2
    mgu_result = MGU(input1[input_dim], input2[input_dim])
    if mgu_result is None:
        print("Failed to unify.")
    else:
        # print("Unification success.")
        sigma, term = mgu_result
        for var, term in sigma.items():
            print(f"  {var} -> {term}")
