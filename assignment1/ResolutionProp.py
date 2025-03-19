# 人工智能 lab1 命题逻辑的归结推理
# 23336020 周子健

from TermParser import parse_term, apply_subst
from MGU import MGU
from RP_components import is_neg, is_diff, to_pos, to_neg

def kb2list(KB: str) -> set:
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
        depth = 0
        las = 0
        c = []
        for i in range(len(clause)):
            if clause[i] == '(':
                depth += 1
            elif clause[i] == ')':
                depth -= 1
            elif depth == 0 and clause[i] == ',':
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
    for i, l in enumerate(C1):
        for j, r in enumerate(C2):
            if is_diff(l, r):
                mgu_result = MGU(to_pos(l), to_pos(r))
                if mgu_result is None:
                    continue
                sigma, term = mgu_result

                def apply_sigma(lit: str) -> str:
                    sign = '~' if is_neg(lit) else ''
                    return sign + repr(apply_subst(parse_term(to_pos(lit)), sigma))
                # add_neg = lambda x: '~' if is_neg(x) else ''
                # apply_sigma = lambda x: add_neg(x) + apply_subst(parse_term(to_pos(x)), sigma).__repr__()
                C1 = [apply_sigma(lit) for lit in C1]
                C2 = [apply_sigma(lit) for lit in C2]
                if term in C1 and to_neg(term) in C2:
                    C1.remove(term)
                    C2.remove(to_neg(term))
                else:
                    C1.remove(to_neg(term))
                    C2.remove(term)
                resolvent = []
                [resolvent.append(l) for l in C1 if l not in resolvent]
                [resolvent.append(r) for r in C2 if r not in resolvent]
                return i, j, sigma, resolvent
    return None


def backtrace(steps, index, chain):
    """
    递归回溯步骤链，将产生空子句证明过程中涉及的步骤索引加入 chain 集合中。
    """
    if index in chain:
        return
    chain.add(index)
    if steps[index]['parents'] is not None:
        for parent in steps[index]['parents']:
            backtrace(steps, parent, chain)


def ResolutionProp(KB: str):
    """Resolution for propositional logic.
    Args:
        KB: str, knowledge base
    Returns:
        list of dict, steps of resolution
    """
    clauses = kb2list(KB)
    steps = [{'parents': None, 'clauses': None, 'sigma': None,
              'resolvent': clause} for clause in clauses]
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

                new_step = {
                    'parents': [i, j],
                    'clauses': [chr(k1 + 97) if len(C1) > 1 else '',
                                chr(k2 + 97) if len(C2) > 1 else ''],
                    'sigma': sigma,
                    'resolvent': resolvent
                }
                new_steps.append(new_step)

                if resolvent == []: # Find resolution
                    # Backtrace to find all steps involved in the proof
                    steps.append(new_step)
                    empty_index = len(steps) - 1
                    chain_indices = set(range(len(clauses)))
                    backtrace(steps, empty_index, chain_indices)
                    chain_indices = sorted(chain_indices)
                    proof_steps = [steps[idx] for idx in chain_indices]

                    # Change parent index
                    for step in proof_steps:
                        if step['parents'] is not None:
                            step['parents'] = [
                                chain_indices.index(p) for p in step['parents']]

                    return proof_steps
        steps.extend(new_steps)
        if not new_steps:
            return None


if __name__ == "__main__":
    KB = [
        '{(firstgrade,),(~firstgrade,child),(~child,)}',
        '{(GradStudent(sue),),(~GradStudent(x),Student(x)),'
        '(~Student(x),~HardWorker(x)),(HardWorker(sue),)}',
        '{(A(tony),),(A(mike),),(A(john),),(L(tony,rain),),'
        '(L(tony,snow),),(~A(x),S(x),C(x)),(~C(y),~L(y,rain)),'
        '(L(z,snow),~S(z)),(~L(tony,u),~L(mike,u)),'
        '(L(tony,v),L(mike,v)),(~A(w),~C(w),S(w))}',
        '{(On(tony,mike),),(On(mike,john),),(Green(tony),),(~Green(john),),'
        '(~On(xx,yy),~Green(xx),Green(yy))}'
    ]
    steps = ResolutionProp(KB[3])
    if steps is None:
        print('No proof found.')
    else:
        for k, step in enumerate(steps):
            out_str = str(k+1)
            if step['parents']:
                out_str += ' ({})'.format(','.join(
                    [str(step['parents'][k]+1) + step['clauses'][k]
                     for k in range(2)]))
            if step['sigma']:
                sigma_str = ''.join(
                    f"{key}={val}" for key, val in step['sigma'].items())
                out_str += '{' + sigma_str + '}'
            out_str += ' = (' + ','.join(step['resolvent']) + ')'
            print(out_str)

