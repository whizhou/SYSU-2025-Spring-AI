import copy

VARIABLES = {'u', 'v', 'w', 'x', 'y', 'z', 'uu', 'vv', 'ww', 'xx', 'yy', 'zz'}

#######################################
# 1. 数据结构：项（Term）和其子类

class Term:
    pass

class Variable(Term):
    def __init__(self, name):
        self.name = name  # 字符串

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(('Variable', self.name))

class Constant(Term):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Constant) and self.name == other.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(('Constant', self.name))

class Function(Term):
    def __init__(self, name, args):
        self.name = name          # 函数名或谓词名，字符串
        self.args = args          # 参数列表，列表内元素为 Term

    def __eq__(self, other):
        return (isinstance(other, Function) and 
                self.name == other.name and 
                len(self.args) == len(other.args) and
                all(a == b for a, b in zip(self.args, other.args)))

    def __repr__(self):
        return f"{self.name}(" + ", ".join(repr(arg) for arg in self.args) + ")"

    def __hash__(self):
        return hash((self.name, tuple(self.args)))

#######################################
# 2. 表示文字（Literal）的类

class Literal:
    def __init__(self, sign, atom):
        self.sign = sign    # True 表示正文字，False 表示否定文字
        self.atom = atom    # atom 为 Function 对象

    def __repr__(self):
        return ("" if self.sign else "~") + repr(self.atom)

    def __eq__(self, other):
        return isinstance(other, Literal) and self.sign == other.sign and self.atom == other.atom

    def __hash__(self):
        return hash((self.sign, self.atom))

#######################################
# 3. 解析器：将字符串解析为 Term 与 Literal  
# 解析项：如果后面跟括号，则认为是函数或谓词；否则根据约定判断变量还是常量  
def parse_term(s):
    s = s.replace(" ", "")
    index = 0

    def parse():
        nonlocal index
        start = index
        # 读取标识符（由字母和数字组成，这里简单认为只由字母组成）
        while index < len(s) and s[index].isalnum():
            index += 1
        name = s[start:index]
        # 如果下一个字符是 '(' ，则认为是函数或谓词
        if index < len(s) and s[index] == '(':
            index += 1  # 跳过 '('
            args = []
            while True:
                arg = parse()
                args.append(arg)
                if index < len(s) and s[index] == ',':
                    index += 1
                elif index < len(s) and s[index] == ')':
                    index += 1
                    break
                else:
                    break
            return Function(name, args)
        else:
            # 判断：若标识符为单个小写字母，则视为变量；否则为常量
            if name in VARIABLES:
                return Variable(name)
            else:
                return Constant(name)
    term = parse()
    return term

# 解析文字：若以 '~' 开头，则为否定文字
def parse_literal(s):
    s = s.strip()
    if s.startswith("~"):
        atom = parse_term(s[1:])
        return Literal(False, atom)
    else:
        atom = parse_term(s)
        return Literal(True, atom)

#######################################
# 4. MGU 算法及辅助函数
# apply_subst: 将替换应用到项上
def apply_subst(term, subst):
    if isinstance(term, Variable):
        if term.name in subst:
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

# Occurs check：检查变量 var 是否在 term 中出现
def occurs(var, term, subst):
    term = apply_subst(term, subst)
    if isinstance(term, Variable):
        return term.name == var.name
    elif isinstance(term, Function):
        return any(occurs(var, arg, subst) for arg in term.args)
    else:
        return False

# unify：最一般合一算法，输入两个 Term，返回 substitution 字典或 None（合一失败）
def unify(t1, t2):
    equations = [(t1, t2)]
    sigma = {}
    while equations:
        s, t = equations.pop(0)
        s = apply_subst(s, sigma)
        t = apply_subst(t, sigma)
        if s == t:
            continue
        elif isinstance(s, Variable):
            if occurs(s, t, sigma):
                return None
            else:
                sigma[s.name] = t
                # 将该替换作用到剩余方程中
                new_eqs = []
                for l, r in equations:
                    new_eqs.append((apply_subst(l, {s.name: t}), apply_subst(r, {s.name: t})))
                equations = new_eqs
                for var in sigma:
                    sigma[var] = apply_subst(sigma[var], {s.name: t})
        elif isinstance(t, Variable):
            if occurs(t, s, sigma):
                return None
            else:
                sigma[t.name] = s
                new_eqs = []
                for l, r in equations:
                    new_eqs.append((apply_subst(l, {t.name: s}), apply_subst(r, {t.name: s})))
                equations = new_eqs
                for var in sigma:
                    sigma[var] = apply_subst(sigma[var], {t.name: s})
        elif isinstance(s, Function) and isinstance(t, Function):
            if s.name != t.name or len(s.args) != len(t.args):
                return None
            else:
                equations = list(zip(s.args, t.args)) + equations
        else:
            return None
    return sigma

#######################################
# 5. 标准化（Standardize Apart）：对一个字句中出现的变量重命名
def standardize_term(term, mapping, suffix):
    if isinstance(term, Variable):
        if term.name not in mapping:
            mapping[term.name] = term.name + "_" + suffix
        return Variable(mapping[term.name])
    elif isinstance(term, Constant):
        return term
    elif isinstance(term, Function):
        new_args = [standardize_term(arg, mapping, suffix) for arg in term.args]
        return Function(term.name, new_args)
    else:
        return term

def standardize_clause(clause, suffix):
    mapping = {}
    new_literals = []
    for lit in clause.literals:
        new_atom = standardize_term(lit.atom, mapping, suffix)
        new_literals.append(Literal(lit.sign, new_atom))
    clause.literals = new_literals
    return clause

#######################################
# 6. 字句（Clause）类，包含归结过程中附带的信息
class Clause:
    def __init__(self, literals, derivation_info=None, id=None):
        self.literals = literals  # list of Literal
        self.derivation_info = derivation_info  # 可记录初始字句或归结信息
        self.id = id  # 字句编号

    def __str__(self):
        return "(" + ",".join(str(lit) for lit in self.literals) + ")"

    def __repr__(self):
        return self.__str__()

    # 若字句中只有 1 个文字，则用该文字的谓词名作为标记；否则按从 a, b, c, … 顺序标记
    def get_literal_labels(self):
        if len(self.literals) == 1:
            return [self.literals[0].atom.name]
        else:
            labels = []
            for i in range(len(self.literals)):
                labels.append(chr(ord('a')+i))
            return labels

    def get_literal_label(self, literal):
        labels = self.get_literal_labels()
        for i, lit in enumerate(self.literals):
            if lit == literal:
                return labels[i]
        return ""

#######################################
# 7. 辅助函数：对文字应用替换；移除重复文字；生成字句的规范表示
def apply_subst_literal(lit, subst):
    new_atom = apply_subst(lit.atom, subst)
    return Literal(lit.sign, new_atom)

def remove_duplicates(literals):
    seen = set()
    result = []
    for lit in literals:
        rep = str(lit)
        if rep not in seen:
            seen.add(rep)
            result.append(lit)
    return result

def canonical_clause(clause):
    # 对字句中文字按字符串排序后合成规范表示
    lits = sorted([str(lit) for lit in clause.literals])
    return "(" + ",".join(lits) + ")"

#######################################
# 8. 两字句归结：对给定的两个字句尝试所有可能的归结
def resolve(clause1, clause2):
    resolvents = []
    # 枚举 clause1 与 clause2 中的文字对
    for i, lit1 in enumerate(clause1.literals):
        for j, lit2 in enumerate(clause2.literals):
            # 两文字必须互补（一个正一个否）且谓词名、参数个数相同
            if lit1.sign != lit2.sign:
                if isinstance(lit1.atom, Function) and isinstance(lit2.atom, Function):
                    if lit1.atom.name == lit2.atom.name and len(lit1.atom.args) == len(lit2.atom.args):
                        # 尝试合一
                        subst = unify(lit1.atom, lit2.atom)
                        if subst is not None:
                            # 归结：去掉被归结的两个文字，其它文字合并后并施加替换
                            new_literals = []
                            for k, l in enumerate(clause1.literals):
                                if k != i:
                                    new_literals.append(apply_subst_literal(l, subst))
                            for k, l in enumerate(clause2.literals):
                                if k != j:
                                    new_literals.append(apply_subst_literal(l, subst))
                            new_literals = remove_duplicates(new_literals)
                            new_clause = Clause(new_literals)
                            # 记录归结信息：记录父字句编号与所用文字标记，以及合一得到的替换
                            label1 = clause1.get_literal_label(lit1)
                            label2 = clause2.get_literal_label(lit2)
                            info = (clause1.id, label1, clause2.id, label2, subst)
                            resolvents.append((new_clause, info))
    return resolvents

#######################################
# 9. ResolutionProp 函数
# 输入 KB 为 list(list(str))，返回归结步骤列表（每一步为字符串说明）
def ResolutionProp(KB: list):
    steps = []       # 保存每一步归结说明
    clauses = []     # 存放 Clause 对象
    clause_set = {}  # 用于检测重复字句（规范表示 → 编号）
    next_id = 1

    # 解析初始字句，并对每个字句进行标准化（变量重命名，后缀为字句编号）
    for clause_str_list in KB:
        literals = [parse_literal(s) for s in clause_str_list]
        clause = Clause(literals, derivation_info="initial", id=next_id)
        clause = standardize_clause(clause, str(next_id))
        clauses.append(clause)
        step_line = f"{next_id} {clause}"
        steps.append(step_line)
        clause_set[canonical_clause(clause)] = next_id
        next_id += 1

    new_generated = clauses.copy()
    resolved = False

    # 主循环：枚举所有字句对进行归结，若产生新字句则加入
    while True:
        new_clauses = []
        n = len(clauses)
        for i in range(n):
            for j in range(i+1, n):
                # 对于每一对字句尝试归结
                resolvents = resolve(clauses[i], clauses[j])
                for (resolvent, info) in resolvents:
                    canon = canonical_clause(resolvent)
                    if canon not in clause_set:
                        clause_set[canon] = next_id
                        resolvent.id = next_id
                        # 构造归结步骤说明
                        id1, lab1, id2, lab2, subst = info
                        if subst:
                            subst_items = [f"{var}={subst[var]}" for var in subst]
                            subst_str = "{" + ",".join(subst_items) + "}"
                        else:
                            subst_str = ""
                        # 对父字句若只有1个文字，则直接用编号，否则显示“编号+文字标记”
                        parent1 = f"{id1}{lab1}" if len([l for l in clauses[i].literals]) > 1 else f"{id1}"
                        parent2 = f"{id2}{lab2}" if len([l for l in clauses[j].literals]) > 1 else f"{id2}"
                        resolution_line = f"{next_id} R[{parent1},{parent2}]{subst_str} = {resolvent}"
                        steps.append(resolution_line)
                        new_clauses.append(resolvent)
                        next_id += 1
                        # 若产生空字句，则归结成功
                        if len(resolvent.literals) == 0:
                            steps.append(f"{next_id} Derived empty clause.")
                            return steps
        if not new_clauses:
            # 没有产生新字句，归结结束
            return steps
        for c in new_clauses:
            clauses.append(c)

def reslove_KB2set(KB: str) -> set:
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

def solute(KB: str):
    S = reslove_KB2set(KB)
    return ResolutionProp(S)
    # print(S)
    # resolutions = [f"({','.join(clause)})" for clause in S]

#######################################
# 10. 测试示例

if __name__ == '__main__':
    # KB 为字句集，每个字句为一个文字列表（字符串形式）
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

    steps = solute(KB[3])
    for step in steps:
        print(step)
