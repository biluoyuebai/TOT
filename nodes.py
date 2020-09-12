class Statement:
    def __init__(self, lhs, rhs, static=False):
        self.rhs = rhs
        self.lhs = lhs
        self.static = static
    def __repr__(self):
        return f'{"static " if self.static else ""}{self.lhs}={self.rhs}'

class ForRangeBlock:
    def __init__(self, var, range):
        self.var = var
        self.range = range
        self.body = []
    def __repr__(self):
        return f'for {self.var} in {self.range}:\n{self.body}'

class ForEachBlock:
    def __init__(self, var, tensor, ind):
        self.var = var
        self.tensor = tensor
        self.ind = ind

class IndexSlice:
    def __init__(self, low, high, step=None):
        self.low = low
        self.high = high
        self.step = step

class Return:
    def __init__(self, retvals):
        self.retvals = retvals

class Module:
    def __init__(self, name):
        self.name = name
        self.args = []
        self.pars = []
        self.body = []
        self.retvals = []
    def __repr__(self):
        retval = [self.name]
        if len(self.args):
            retval += '['
        for arg in self.args:
            retval += arg.name
            if arg.default is not None:
                retval += '='
                retval += str(arg.default)
            retval += ','
        if len(self.args):
            retval[-1] = ']'
        if len(self.pars):
            retval += '('
        for par in self.pars:
            retval += par.name
            retval += ','
        if len(self.pars):
            retval[-1] = ')'
        return ''.join(retval)

class ModuleRef:
    def __init__(self, ref, alias, args, pars):
        self.alias = alias
        self.ref = ref if isinstance(ref, str) else ref.name
        self.args = args
        self.pars = pars
    def __repr__(self):
        retval = [self.ref]
        if self.alias:
            retval += '_'
            retval += self.alias
        if len(self.args):
            retval += '['
        for arg in self.args:
            retval += arg if isinstance(arg, str) else repr(arg)
            retval += ','
        if len(self.args):
            retval[-1] = ']'
        if len(self.pars):
            retval += '('
        for par in self.pars:
            retval += par if isinstance(par, str) else repr(par)
            retval += ','
        if len(self.pars):
            retval[-1] = ')'
        return ''.join(retval)

class Argument:
    def __init__(self, name, type=None, default=None):
        self.name = name
        self.type = type
        self.default = default

class Parameter:
    def __init__(self, name, type=None, shape=['...'], default=None):
        self.name = name
        self.type = type
        self.shape = shape
        self.default = default

class VaPar(Parameter):
    def __init__(self, type=None, shape=None):
        super(VaPar, self).__init__('va_par', type, shape)

class VaArg(Argument):
    def __init__(self, type=None, default=None):
        super(VaArg, self).__init__('va_arg', type, default)

class Retval:
    def __init__(self, name, type=None, shape=['...']):
        self.name = name
        self.type = type
        self.shape = shape

class LocalVar:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name

class IfElseBlock:
    def __init__(self):
        self.conds = []
        self.bodies = []
    def __repr__(self):
        return repr(list(zip(self.conds, self.bodies)))

class Identifier:
    def __init__(self, tokens):
        self.name = ''.join(tokens)
    def __repr__(self):
        return self.name