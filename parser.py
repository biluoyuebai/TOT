import pyparsing as pp
from pyparsing import pyparsing_common as ppc
from builtin_modules import builtins
from nodes import *

import sys

class Parser:
    def __init__(self):
        self.build_bnfs()
    def build_bnfs(self):
        self.float = ppc.real
        self.int = ppc.integer
        self.number = ppc.number
        self.wildcard = pp.Empty().setParseAction(lambda _: ':')
        self.rawstr = pp.oneOf('"\'') + pp.Word(pp.alphanums) + pp.oneOf('"\'')
        self.ident = pp.Char('_') | (pp.Word(pp.alphas, pp.alphanums+'_') + pp.Optional(pp.Char('~') + pp.Word(pp.alphanums)) + pp.Optional(pp.Char('^') + pp.Word(pp.alphanums)))
        self.ident = self.ident.setParseAction(self._ident)
        self.expr = pp.Forward()
        self.atom = pp.Forward()
        self.slice = (self.expr | self.wildcard) + ':' + (self.expr | self.wildcard) + pp.Optional(':' + (self.expr | self.wildcard)) # in-dev v0.0.2
        self.slice = self.slice.setParseAction(self._slice)
        self.varindex = self.ident + pp.Suppress('[') + pp.Group(pp.delimitedList(self.slice | self.expr | ':' | self.wildcard)) + pp.Suppress(']')# + pp.Optional(pp.Suppress('(') + pp.Group(pp.delimitedList(self.expr)) + pp.Suppress(')'))
        self.varindex = self.varindex.setParseAction(self._varindex)
        self.arbitraryfncall = (self.ident + pp.Optional(pp.Suppress('{') + pp.Word(pp.alphanums + '@', pp.alphanums + '_') + pp.Suppress('}')) + pp.Optional(pp.Suppress('[') + pp.Group(pp.delimitedList(self.expr | ':' | self.wildcard)) + pp.Suppress(']')) +\
            (pp.Suppress('(') + pp.Group(pp.delimitedList(self.expr)) + pp.Suppress(')'))).setParseAction(self._fncall)
        self.unaryfncall_ = pp.Forward()
        self.unaryfncall_tail_ = pp.Group(pp.Suppress('.') + self.ident + pp.Optional(pp.Suppress('{') + pp.Word(pp.alphanums + '@', pp.alphanums) + pp.Suppress('}')) + pp.Optional(pp.Suppress('[') + pp.Group(pp.delimitedList(self.expr | ':' | self.wildcard)) + pp.Suppress(']')))
        self.unaryfncall_head_ = (pp.Suppress('(') + self.expr + pp.Suppress(')')) | self.varindex | self.ident | self.wildcard
        # self.unaryfncall = (self.unaryfncall_head_ + self.unaryfncall_tail_) ^ (self.arbitraryfncall + self.unaryfncall_tail_) + self.unaryfncall_
        # self.unaryfncall = (self.unaryfncall_head_ + self.unaryfncall_tail_) | (self.arbitraryfncall + self.unaryfncall_tail_) + self.unaryfncall_
        self.unaryfncall = (self.arbitraryfncall + self.unaryfncall_tail_) | (self.unaryfncall_head_ + self.unaryfncall_tail_) + self.unaryfncall_
        self.unaryfncall_ <<= (self.unaryfncall_tail_ + self.unaryfncall_) | pp.empty
        self.unaryfncall = self.unaryfncall.setParseAction(self._unaryfncall)
        self.fncall = self.unaryfncall | self.arbitraryfncall
        self.atom <<= self.rawstr | self.number | (pp.oneOf('+ -') + self.atom).setParseAction(self._posneg) | self.fncall | (pp.Suppress('(') + self.expr + pp.Suppress(')')) | self.varindex | self.ident
        self.matmul_ = pp.Forward()
        self.matmul = self.atom + self.matmul_
        self.matmul_ <<= (pp.Suppress('#') + self.atom + self.matmul_) | pp.empty
        self.matmul = self.matmul.setParseAction(self._matmul)
        self.muldiv_ = pp.Forward()
        self.muldiv = self.matmul + self.muldiv_
        self.muldiv_ <<= (pp.oneOf('* / //') + self.matmul + self.muldiv_) | pp.empty
        self.muldiv = self.muldiv.setParseAction(self._muldiv)
        self.addsub_ = pp.Forward()
        self.addsub = self.muldiv + self.addsub_
        self.addsub_ <<= (pp.oneOf('+ -') + self.muldiv + self.addsub_) | pp.empty
        self.addsub = self.addsub.setParseAction(self._addsub)
        self.comp = self.addsub + pp.oneOf('> < == >= <= !=') + self.addsub
        self.comp = self.comp.setParseAction(self._comp)
        self.boolexpr = pp.Forward()
        self.boolatom = pp.Forward()
        self.boolatom <<= (pp.oneOf('not binot ! ~') + self.boolatom).setParseAction(self._not) | self.comp | (pp.Suppress('(') + self.boolexpr + pp.Suppress(')')) | pp.Keyword('True').setParseAction(self._ident) | pp.Keyword('False').setParseAction(self._ident)
        self.booland_ = pp.Forward()
        self.booland = self.boolatom + self.booland_
        self.booland_ <<= (pp.oneOf('and biand && &') + self.boolatom + self.booland_) | pp.empty
        self.booland = self.booland.setParseAction(self._and)
        self.boolor_ = pp.Forward()
        self.boolor = self.booland + self.boolor_
        self.boolor_ <<= (pp.oneOf('or bior || |') + self.booland + self.boolor_) | pp.empty
        self.boolor = self.boolor.setParseAction(self._or)
        self.boolexpr <<= self.boolor
        self.expr <<= self.boolexpr ^ self.addsub
        self.body = pp.Keyword('body')
        self.end = pp.Keyword('end')
        self.for_ = pp.Keyword('for')
        self.in_ = pp.Keyword('in')
        self.if_ = pp.Keyword('if')
        self.elif_ = pp.Keyword('elif')
        self.else_ = pp.Keyword('else')
        self.return_ = pp.Keyword('return')
        self.static = pp.Keyword('static')
        self.forhead = self.for_ + self.ident + pp.Suppress(self.in_) + pp.oneOf('[ (') + pp.Group(pp.delimitedList(self.expr)) + pp.oneOf('] )')
        self.ifhead = self.if_ + self.expr
        self.elifhead = self.elif_ + self.expr
        self.elsehead = self.else_
        self.return_stat = self.return_ + pp.Group(pp.delimitedList(self.expr))
        self.statement = (pp.Optional(self.static) + pp.Group(pp.delimitedList(self.varindex | self.ident)) + pp.Suppress('=') + self.expr) |\
            self.return_stat | self.forhead | self.ifhead | self.elifhead | self.elsehead | self.end
    def parse(self, line):
        return self.statement.parseString(line, True).asList()
    def _slice(self, tokens):
        if len(tokens) == 5:
            return IndexSlice(tokens[0], tokens[2], tokens[4])
        else:
            return IndexSlice(tokens[0], tokens[2])
    def _ident(self, tokens):
        return Identifier(tokens)
    def _unaryfncall(self, tokens):
        # print('unary', tokens, len(tokens))
        head = tokens[0]
        if head == ':':
            pars = []
        else:
            pars = [head]
        for tail in tokens[1:]:
            if len(tail) == 3:
                head = ModuleRef(tail[0], tail[1], tail[2], pars)
            elif len(tail) == 2:
                if isinstance(tail[1], str):
                    head = ModuleRef(tail[0], tail[1], [], pars)
                else:
                    head = ModuleRef(tail[0], None, tail[1], pars)
            elif len(tail) == 1:
                head = ModuleRef(tail[0], None, [], pars)
            pars = [head]
        return head
    def _fncall(self, tokens):
        # print('arbitrary', tokens)
        # if tokens[0].name[0] == 'S':
            # print(tokens)
        if len(tokens) == 4:
            # print(tokens[1])
            return ModuleRef(tokens[0], tokens[1], tokens[2], tokens[3])
        elif len(tokens) == 3:
            if isinstance(tokens[1], str):
                return ModuleRef(tokens[0], tokens[1], [], tokens[2])
            else:
                return ModuleRef(tokens[0], None, tokens[1], tokens[2])
        elif len(tokens) == 2:
            return ModuleRef(tokens[0], None, [], tokens[1])
    def _posneg(self, tokens):
        if tokens[0] == '+':
            return tokens[1]
        else:
            return ModuleRef('neg', None, [], [tokens[1]])
    def _muldiv(self, tokens):
        if len(tokens) == 1:
            return tokens[0]
        else:
            lhs = tokens[0]
            for op, rhs in zip(tokens[1::2], tokens[2::2]):
                lhs = ModuleRef(op, None, [], [lhs, rhs])
            return lhs
    def _matmul(self, tokens):
        if len(tokens) == 1:
            return tokens[0]
        else:
            lhs = tokens[0]
            for rhs in tokens[1:]:
                lhs = ModuleRef('matmul', None, [], [lhs, rhs])
            return lhs
    def _addsub(self, tokens):
        if len(tokens) == 1:
            return tokens[0]
        else:
            lhs = tokens[0]
            for op, rhs in zip(tokens[1::2], tokens[2::2]):
                lhs = ModuleRef(op, None, [], [lhs, rhs])
            return lhs
    def _varindex(self, tokens):
        var = tokens[0]
        inds = tokens[1]
        assert len(inds) != 0, 'please assure that at least one dim is in indices!'
        return ModuleRef('[]', None, [], [var] + inds.asList())
    def _comp(self, tokens):
        return ModuleRef(tokens[1], None, [], [tokens[0], tokens[-1]])
    def _and(self, tokens):
        if len(tokens) == 1:
            return tokens[0]
        else:
            lhs = tokens[0]
            for op, rhs in zip(tokens[1::2], tokens[2::2]):
                lhs = ModuleRef(op, None, [], [lhs, rhs])
            return lhs
    def _or(self, tokens):
        if len(tokens) == 1:
            return tokens[0]
        else:
            lhs = tokens[0]
            for op, rhs in zip(tokens[1::2], tokens[2::2]):
                lhs = ModuleRef(op, None, [], [lhs, rhs])
            return lhs
    def _not(self, tokens):
        return ModuleRef(tokens[0], None, [], [tokens[1]])

class ParserContext:
    def __init__(self):
        self.modus = {}
        self.thirdparty = {}
        self.this_modu = None
        self.env_stack = []
        self.parser = None
        self.locals = {'None': LocalVar('None'), 'True': LocalVar('True'), 'False': LocalVar('False')}
        self.modus.update(builtins)
    def rm_modu(self, name):
        assert name not in builtins, "you can not remove a builtin module"
        del self.modus[name]
    def add_thirdparty_modu(self, name, argnames, argtypes, argdefaults, parnames, partypes, parshapes, retvals):
        assert name not in self.modus, f'overlapped module name {name}!'
        modu = Module(name)
        for argname, argtype, argdefault in zip(argnames, argtypes, argdefaults):
            if argname != ':':
                modu.args.append(Argument(argname, argtype, argdefault))
            else:
                modu.args.append(VaArg(argtype, argdefault))
        for parname, partype, parshape in zip(parnames, partypes, parshapes):
            if parname != ':':
                modu.pars.append(Parameter(parname, partype, parshape))
            else:
                modu.pars.append(VaPar(partype, parshape))
        modu.retvals.extend(retvals)
        self.modus[name] = modu
        self.thirdparty[name] = modu
    def rm_thirdparty_modu(self, name):
        del self.thirdparty[name]
        del self.modus[name]
    def init_modu(self, name):
        assert name not in self.modus, 'module name repeated!'
        self.this_modu = Module(name)
    def revise_modu(self, name):
        assert name not in builtins and name not in self.thirdparty, f"have no authority to revise {name}"
        self.this_modu = self.modus[name]
        del self.modus[name]
    def save_modu(self):
        self.modus[self.this_modu.name] = self.this_modu
        self.this_modu = None
        self.env_stack = []
        self.locals = {'None': LocalVar('None'), 'True': LocalVar('True'), 'False': LocalVar('False')}
    def abandon_modu(self):
        self.this_modu = None
    def add_arg(self, name, dtype, default=None):
        assert self.parser is None, "please add args before parser was launched!"
        self.this_modu.args.append(Argument(name, dtype, default))
    def add_par(self, name, dtype, shape, default=None):
        assert self.parser is None, "please add pars before parser was launched!"
        self.this_modu.pars.append(Parameter(name, dtype, shape, default))
    def rm_arg(self, ind):
        assert self.parser is None, "please rm args before parser was launched!"
        del self.this_modu.args[ind]
    def rm_par(self, ind):
        assert self.parser is None, "please rm pars before parser was launched!"
        del self.this_modu.args[ind]
    def add_retval(self, name, type, shape=['...'], name2=None, type2=None, shape2=None, name3=None, type3=None, shape3=None):
        retvals = [Retval(name, type, shape)]
        if name2 is not None:
            retvals.append(Retval(name2, type2, shape2))
            if name3 is not None:
                retvals.append(Retval(name3, type3, shape3))
        self.this_modu.retvals.append(retvals)
    def add_retvals(self, l):
        self.this_modu.retvals.append(l)
    def rm_retval(self, ind):
        del self.this_modu.retvals[ind]
    def launch_parser(self):
        self.parser = Parser()
        for arg in self.this_modu.args:
            self.locals[arg.name] = LocalVar(arg.name)
        for par in self.this_modu.pars:
            self.locals[par.name] = LocalVar(par.name)
    def kill_parser(self, abandon=True):
        self.parser = None
    def static_check_identifier(self, node):
        # print(self.locals)
        assert node.name in self.locals, f'unknown identifier name {node.name}'        
    def static_check_moduleref(self, node):
        assert node.ref in self.modus, f'unknown module name {node.ref}'
        target = self.modus[node.ref]
        i = len(target.args) - 1
        if i >= 0 and isinstance(target.args[i], VaArg):
            if target.args[-1].default is None:
                for arg in node.args[i:]:
                    assert arg != ':', f'VaArg has no default value, in {node.ref}'
            i -= 1
        while i >= 0 and target.args[i].default is not None:
            i -= 1
        assert len(node.args) > i, f'too few args are given, to {node.ref}'
        i += 1
        for arg, tar in zip(node.args[:i], target.args[:i]):
            if isinstance(arg, str) and arg == ':':
                assert tar.default is not None, f'omitted args must have default values, in {node.ref}, {tar.name}'
                
        for arg in node.args:
            if isinstance(arg, ModuleRef):
                argtarget = self.static_check_moduleref(arg)
                for retval in argtarget.retvals:
                    assert len(retval) == 1, f'multiple retvals are assigned to a single arg, in {node.ref}, {tar.name}'
            elif isinstance(arg, Identifier):
                self.static_check_identifier(arg)
#         if isinstance(target.pars[-1], VaPar):
#             assert len(target.pars) - 1 <= len(node.pars), f'at least {len(target.pars) - 1} pars are needed, in {node.ref}'
        for par in node.pars:
            if isinstance(par, ModuleRef):
                partarget = self.static_check_moduleref(par)
                for retval in partarget.retvals:
                    assert len(retval) == 1, f'multiple retvals are assigned to a single par, in {node.ref}, {par.name}'
            elif isinstance(par, Identifier):
                self.static_check_identifier(par)
        return target
        
    def parse(self, line):
        tokens = self.parser.parse(line)
        if isinstance(tokens[0], str):
            keyword = tokens[0]
            if keyword == 'if':
                if isinstance(tokens[1], ModuleRef):
                    self.static_check_moduleref(tokens[1])
                else:
                    self.static_check_identifier(tokens[1])
                self.env_stack.append(IfElseBlock())
                self.env_stack[-1].conds.append(tokens[1])
                self.env_stack[-1].bodies.append([])
            elif keyword == 'elif':
                assert isinstance(self.env_stack[-1], IfElseBlock), 'syntax error with elif'
                if isinstance(tokens[1], ModuleRef):
                    self.static_check_moduleref(tokens[1])
                else:
                    self.static_check_identifier(tokens[1])
                self.env_stack[-1].conds.append(tokens[1])
                self.env_stack[-1].bodies.append([])
            elif keyword == 'else':
                assert isinstance(self.env_stack[-1], IfElseBlock), 'syntax error with elif'
                self.env_stack[-1].conds.append(None)
                self.env_stack[-1].bodies.append([])
            elif keyword == 'for':
                assert tokens[1].name not in set(map(lambda a: a.name, self.this_modu.args)), f'args are read only, with {tokens[1].name}'
                var = LocalVar(tokens[1].name)
                if len(tokens[3]) == 3:
                    range_limit = tokens[3]
                elif len(tokens[3]) == 1:
                    range_limit = [0 + tokens[3][0] + 1]
                else:
                    range_limit = tokens[3] + [1]
                range = ModuleRef('range', None, [], range_limit)
                self.locals[tokens[1]] = var
                if isinstance(tokens[2], str) and tokens[2] == '[':
                    for tok in tokens[3]:
                        if isinstance(tok, ModuleRef):
                            self.static_check_moduleref(tok)
                        elif isinstance(tok, Identifier):
                            self.static_check_identifier(tok)
                    self.env_stack.append(ForRangeBlock(var, range))
                else:
                    raise NotImplementedError
            elif keyword == 'end':
                self.this_modu.body.append(self.env_stack[-1])
                self.env_stack.pop()
            elif keyword == 'return':
                for retval in tokens[1]:
                    if isinstance(retval, ModuleRef):
                        self.static_check_moduleref(retval)
                    else:
                        self.static_check_identifier(retval)
                if len(self.env_stack) == 0:
                    self.this_modu.body.append(Return(tokens[1]))
                elif isinstance(self.env_stack[-1], IfElseBlock):
                    self.env_stack[-1].bodies[-1].append(Return(tokens[1]))
                elif isinstance(self.env_stack[-1], ForRangeBlock):
                    self.env_stack[-1].body.append(Return(tokens[1]))
            elif keyword == 'static':
                lhs = []
                for ident in tokens[1]:
                    if isinstance(ident, Identifier):
                        assert ident.name not in set(map(lambda a: a.name, self.this_modu.args)), f'args are read only, with {ident.name}'
                        local = LocalVar(ident.name)
                        lhs.append(local)
                        self.locals[ident.name] = local
                    else:
                        self.static_check_moduleref(ident)
                        lhs.append(ident)
                self.static_check_moduleref(tokens[2])
                if len(self.env_stack) == 0:
                    self.this_modu.body.append(Statement(lhs, tokens[-1], True))
                else:
                    raise NotImplementedError('static statements can not be compound blocks.')
            else:
                raise RuntimeError(f'syntax error with {keyword}')
        else:
            lhs = []
            for ident in tokens[0]:
                if isinstance(ident, Identifier):
                    assert ident.name not in set(map(lambda a: a.name, self.this_modu.args)), f'args are read only, with {ident.name}'
                    local = LocalVar(ident.name)
                    lhs.append(local)
                    self.locals[ident.name] = local
                else:
                    self.static_check_moduleref(ident)
                    lhs.append(ident)
            self.static_check_moduleref(tokens[1])
            if len(self.env_stack) == 0:
                self.this_modu.body.append(Statement(lhs, tokens[-1]))
            elif isinstance(self.env_stack[-1], IfElseBlock):
                self.env_stack[-1].bodies[-1].append(Statement(lhs, tokens[-1]))
            elif isinstance(self.env_stack[-1], ForRangeBlock):
                self.env_stack[-1].body.append(Statement(lhs, tokens[-1]))
    def generate(self, generator):
        return generator.generate(self.this_modu, self.modus)
def file_to_file(in_file, out_file, generator):
    pc = ParserContext()
    with open(in_file, 'r') as reader:
        lines = reader.readlines()
    output = []    
    i = 0
    while i < len(lines):
        if lines[i][:6] == 'module':
            pc.init_modu(lines[i][7:-1])
            i += 1
            while lines[i][:3] == 'arg':
                arg_disc = [s.strip() for s in lines[i][4:].split('|')]
                if len(arg_disc) == 2:
                    pc.add_arg(arg_disc[0], arg_disc[1])
                else:
                    pc.add_arg(arg_disc[0], arg_disc[1], arg_disc[2])
                i += 1
                # with open('log.txt', 'a') as writer:
                    # writer.write(f'{arg_disc[0]}\n')
            while lines[i][:3] == 'par':
                par_disc = [s.strip() for s in lines[i][4:].split('|')]
                if len(par_disc) == 2:
                    pc.add_par(par_disc[0], par_disc[1], ['...'])
                elif len(par_disc) == 3:
                    shape = [s.strip() for s in par_disc[-1].split(',')]
                    pc.add_par(par_disc[0], par_disc[1], shape)
                else:
                    shape = [s.strip() for s in par_disc[-1].split(',')]
                    pc.add_par(par_disc[0], par_disc[1], shape, par_disc[3])
                i += 1
                # with open('log.txt', 'a') as writer:
                    # writer.write(f'{par_disc[0]}\n')
            retvals = []
            while lines[i][:3] == 'ret':
                ret_disc = [s.strip() for s in lines[i][4:].split('|')]
                if len(ret_disc) == 2:
                    retvals.extend([ret_disc[0], ret_disc[1], ['...']])
                elif len(par_disc) == 3:
                    shape = [s.strip() for s in par_disc[-1].split(',')]
                    retvals.extend([ret_disc[0], ret_disc[1], shape])
                i += 1
                # with open('log.txt', 'a') as writer:
                    # writer.write(f'{ret_disc[0]}\n')
            if retvals:
                pc.add_retval(*retvals)
            if lines[i][:4] == 'body' or lines[i][:7] == 'forward':
                i += 1
            pc.launch_parser()
            while lines[i][:9] != 'moduleend':
                # with open('log.txt', 'a') as writer:
                    # writer.write(f'{lines[i]}')
                cur_line = []
                line = lines[i].strip()
                if len(line) and line[0] == '#':
                    i += 1
                    continue
                while len(line) == 0 or line[-1] == '\\' or line[0] == '#':
                    if len(line) and line[0] == '#':
                        i += 1
                        continue
                    cur_line.append(line[:-1])
                    i += 1
                    line = lines[i].strip()
                
                cur_line.append(line)
                cur_line = ''.join(cur_line)
                # print(cur_line)
                # a = input()
                try:
                    pc.parse(cur_line)
                except Exception as e:
                    sys.stderr.write('in ')
                    sys.stderr.write(cur_line)
                    sys.stderr.write('\n')
                    raise e
                i += 1
            i += 1
            output.append(pc.generate(generator))
            output.append('\n' * 2)
            pc.save_modu()
            pc.kill_parser()
        elif lines[i] == '\n':
            i += 1
        else:
            raise RuntimeError(f'Syntax Error in {in_file}, line {i}, sentence {lines[i]}')
        with open(out_file, 'w') as writer:
            for pymodu in generator.python_modules.values():
                writer.write(pymodu)
                writer.write('\n')
            writer.writelines(['\n' * 2])
            writer.writelines(output)