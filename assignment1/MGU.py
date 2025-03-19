# 定义全局变量集合，规定哪些标识符为变量
VARIABLES = {'u', 'v', 'w', 'x', 'y', 'z', 'uu', 'vv', 'ww', 'xx', 'yy', 'zz'}

# 定义项的数据结构

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

# 解析器：将输入字符串转换成项
def parse_term(s):
    # 去除空白字符
    s = s.replace(" ", "")
    index = 0

    def parse():
        nonlocal index
        # 读取标识符 exist of characters
        start = index
        while index < len(s) and s[index].isalpha():
            index += 1
        name = s[start:index]
        # 如果下一个字符是'('，则认为是函数/谓词
        if index < len(s) and s[index] == '(':
            index += 1  # 跳过'('
            args = []
            while True:
                arg = parse()
                args.append(arg)
                if index < len(s) and s[index] == ',':
                    index += 1  # 跳过','
                elif index < len(s) and s[index] == ')':
                    index += 1  # 跳过')'
                    break
                else:
                    break
            return Function(name, args)
        else:
            # 如果标识符在变量集合中，则认为是变量，否则作为常量处理
            if name in VARIABLES:
                return Variable(name)
            else:
                return Constant(name)
    
    term = parse()
    return term

# 应用替换，将 substitution（dict）作用于项上
def apply_subst(term, subst):
    if isinstance(term, Variable):
        if term.name in subst:
            # 递归应用替换，防止替换项中还含有变量
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

# Occurs check：检查变量 var 是否在 term 内部出现（经过当前替换）
def occurs(var, term, subst):
    term = apply_subst(term, subst)
    if isinstance(term, Variable):
        return term.name == var.name
    elif isinstance(term, Function):
        return any(occurs(var, arg, subst) for arg in term.args)
    else:
        return False

# 最一般合一算法
def unify(t1, t2):
    # E 为方程列表，sigma 为替换字典（变量名 -> Term）
    equations = [(t1, t2)]
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

# 主接口函数，输入两个字符串，返回最一般合一替换
def MGU(str1, str2):
    term1 = parse_term(str1)
    term2 = parse_term(str2)
    result = unify(term1, term2)
    if result is None:
        return None, None
    term1 = apply_subst(term1, result)
    term2 = apply_subst(term2, result)
    assert term1.__repr__() == term2.__repr__()
    return result, term1.__repr__()


# 测试示例
if __name__ == '__main__':
    input1 = ['P(xx, a)', 'P(a,xx, f(g(yy)))', 'P(a,x,h(g(z)))', 'first']
    input2 = ['P(b, yy)', 'P(zz, f(zz),f(uu))', 'P(z,h(y),h(y))', '~first']
    input_dim = 2
    mgu_result, term = MGU(input1[input_dim], input2[input_dim])
    if mgu_result is None:
        print("合一失败")
    else:
        print("最一般合一替换为：")
        for var, term in mgu_result.items():
            print(f"  {var} -> {term}")
