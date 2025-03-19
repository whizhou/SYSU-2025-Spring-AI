# 人工智能 lab1 命题逻辑的归结推理
# 23336020 周子健

from TermParser import parse_term, apply_subst
from MGU import MGU
from RP_components import is_neg, is_diff, to_pos, to_neg

def reslove_KB2set(KB: str) -> set:
    """Convert knowledge base string to a set of clauses.
    Args:
        KB: str, knowledge base string
    Returns:
        clauses: list of list of str
    """
    KB = KB[2:-2].split('),(')
    KB = [clause.strip(',') for clause in KB]
    clauses = []
    for clause in KB:
        num = 0
        las = 0
        c = []
        for i in range(len(clause)):
            if clause[i] == '(':
                num += 1
            elif clause[i] == ')':
                num -= 1
            elif num == 0 and clause[i] == ',':
                c.append(clause[las:i])
                las = i + 1
        c.append(clause[las:])
        clauses.append(c)
    return clauses

def resolve(C1: list, C2: list) -> list:
    """Resolve two clauses C1 and C2.
    Args:
        C1: list of str
        C2: list of str
    Returns:
        tuple, (int, int, dict, list)
        or None if cannot resolve
    """
    for k1, l in enumerate(C1):
        for k2, r in enumerate(C2):
            if is_diff(l, r):
                mgu_result = MGU(to_pos(l), to_pos(r))
                if mgu_result is None:
                    continue
                sigma, term = mgu_result

                add_neg = lambda x: '~' if is_neg(x) else ''
                apply_sigma = lambda x: add_neg(x) + apply_subst(parse_term(to_pos(x)), sigma).__repr__()
                C1 = [apply_sigma(c) for c in C1]
                C2 = [apply_sigma(c) for c in C2]
                if term in C1 and to_neg(term) in C2:
                    C1.remove(term)
                    C2.remove(to_neg(term))
                else:
                    C1.remove(to_neg(term))
                    C2.remove(term)
                C = []
                [C.append(l) for l in C1 if l not in C]
                [C.append(r) for r in C2 if r not in C]
                return k1, k2, sigma, C
    return None

def ResolutionProp(KB: str):
    """Resolution for propositional logic.
    Args:
        KB: str, knowledge base
    Returns:
        list of dict, steps of resolution
    """
    clauses = reslove_KB2set(KB)
    steps = [{'clauses': None, 'sigma': None, 'resolvent': clause} for k, clause in enumerate(clauses)]
    # S = set(tuple(clause) for clause in clauses)
    while True:
        new_steps = []
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                C1 = steps[i]['resolvent']
                C2 = steps[j]['resolvent']
                res = resolve(C1, C2)
                if res is None:
                    continue
                k1, k2, sigma, resolvent = res
                if resolvent in [step['resolvent'] for step in steps]:
                    continue

                new_step = {'clauses': [], 'sigma': sigma, 'resolvent': resolvent}
                if len(steps[i]['resolvent']) > 1:
                    new_step['clauses'].append(str(i+1) + chr(k1+97))
                else:
                    new_step['clauses'].append(str(i+1))
                if len(steps[j]['resolvent']) > 1:
                    new_step['clauses'].append(str(j+1) + chr(k2+97))
                else:
                    new_step['clauses'].append(str(j+1))
                new_steps.append(new_step)
                # new_steps.append({'clauses': [str(i+1) + chr(k1+65), str(j+1) + chr(k2+65)], 'sigma': sigma, 'resolvent': resolvent})
                if resolvent == []:
                    return steps + new_steps
        steps += new_steps
        if new_steps == []:
            return steps
    return steps


if __name__ == "__main__":
    KB = []
    KB.append('{(firstgrade,),(~firstgrade,child),(~child,)}')

    KB.append('{(GradStudent(sue),),(~GradStudent(x),Student(x)),' + \
        '(~Student(x),~HardWorker(x)),(HardWorker(sue),)}')

    KB.append('{(A(tony),),(A(mike),),(A(john),),(L(tony,rain),),' + \
        '(L(tony,snow),),(~A(x),S(x),C(x)),(~C(y),~L(y,rain)),' + \
        '(L(z,snow),~S(z)),(~L(tony,u),~L(mike,u)),' + \
        '(L(tony,v),L(mike,v)),(~A(w),~C(w),S(w))}')

    KB.append('{(On(tony,mike),),(On(mike,john),),(Green(tony),),' + \
        '(~Green(john),),(~On(xx,yy),~Green(xx),Green(yy))}')

    steps = ResolutionProp(KB[2])
    for k, step in enumerate(steps):
        out_str = str(k+1)
        if step['clauses']:
            out_str += ' R[' + ','.join(step['clauses']) + ']'
        if step['sigma']:
            out_str += '{'
            for key, val in step['sigma'].items():
                out_str += key + '=' + str(val)
            out_str += '}'
        out_str += ' = (' + ','.join(step['resolvent']) + ')'
        print(out_str)

