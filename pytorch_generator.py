from typing import Dict, List, Tuple
from builtin_modules import *
from nodes import *

class PyTorchGenerator:
    def __init__(self, indent='    '):
        self.indent = indent
        self.nn_map = {
            'linear': 'Linear',
            'conv1d': 'Conv1d',
            'conv2d': 'Conv2d',
            'conv3d': 'Conv3d',
            'maxpool1d': 'MaxPool1d',
            'maxpool2d': 'MaxPool2d',
            'maxpool3d': 'MaxPool3d',
            'avgpool1d': 'AvgPool1d',            
            'avgpool2d': 'AvgPool2d',
            'avgpool3d': 'AvgPool3d',
            'batchnorm1d': 'BatchNorm1d',
            'batchnorm2d': 'BatchNorm2d',
            'batchnorm3d': 'BatchNorm3d',
        }
        self.builtin_map = {
            'shape': self._gen_shape,
            'size': self._gen_shape,
            'range': self._gen_range,
            'arange': self._gen_arange,
            '+': self._gen_add,
            '-': self._gen_sub,
            '*': self._gen_mul,
            '/': self._gen_div,
            '//': self._gen_div,
            'T': self._gen_transpose,
            'transpose': self._gen_transpose,
            'neg': self._gen_neg,
            'not': self._gen_not,
            'and': self._gen_and,
            'or': self._gen_or,
            'biand': self._gen_biand,
            'bior': self._gen_bior,
            '&&': self._gen_tand,
            '||': self._gen_tor,
            '!': self._gen_tnot,
            '&': self._gen_tbiand,
            '|': self._gen_tbior,
            '~': self._gen_tbnot,
            'matmul': self._gen_mm,
            'reshape': self._gen_reshape,
            'dropout': self._gen_dropout,
            'cat': self._gen_cat,
            'concatenate': self._gen_cat,
            'stack': self._gen_stack,
            'squeeze': self._gen_squeeze,
            'unsqueeze': self._gen_squeeze,
            'repeat': self._gen_repeat,
            'expand': self._gen_expand,
            'sort': self._gen_sort,
            '[]': self._gen_index,
            'layernorm': self._gen_layernorm
        }
        for k in {'abs', 'sign', 'sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'log10', 'log2'}:
            self.builtin_map[k] = self._gen_unary
        for k in {'>', '<', '>=', '<=', '==', '!='}:
            self.builtin_map[k] = self._gen_comp
        for k in {'arcsin', 'arccos', 'arctan'}:
            self.builtin_map[k] = self._gen_arc
        for k in {'sigmoid', 'tanh', 'softmax', 'softmin', 'relu', 'leakyrelu'}:
            self.builtin_map[k] = self._gen_nonlinear
        for k in {'sabs', 'ssign', 'ssqrt', 'ssin', 'scos', 'stan', 'slog', 'sexp', 'sceil', 'sfloor'}:
            self.builtin_map[k] = self._gen_scaler
        for k in {'sum', 'prod', 'max', 'min', 'argmax', 'argmin', 'mean', 'median', 'std', 'var'}:
            self.builtin_map[k] = self._gen_reduce
        for k in {'zeros', 'ones', 'empty', 'randn'}:
            self.builtin_map[k] = self._gen_consts
        for k in self.nn_map.keys():
            self.builtin_map[k] = self._gen_nns
        self.python_modules = {'torch': 'import torch', 'nn': 'import torch.nn as nn'}
    def generate(self, module : Module, modus : Dict[str, Module]):
        self.modus = modus
        class_header = f"class {module.name}(nn.Module):"
        init_header = self.indent + 'def __init__(self, {}):'
        forward_header = self.indent + 'def forward(self, {}):'
        self.members = {}  # python 3.6+ keep keys' inserting order
        arglist = []
        argdiscs = []
        for arg in module.args:
            argstr = arg.name.replace('~', '_').replace('^', '_')
            if arg.default is not None:
                argstr = argstr + '=' + arg.default
            arglist.append(argstr)
            argdisc = self.indent * 3 + arg.name.replace('~', '_').replace('^', '_') + ': '
            if arg.type is not None:
                argdisc = argdisc + 'type - ' + arg.type
            if arg.default is not None:
                argdisc = argdisc + '; default - ' + arg.default
            argdiscs.append(argdisc)
            self.members[arg.name.replace('~', '_').replace('^', '_')] =\
                (self.indent * 2 + 'self.' + arg.name.replace('~', '_').replace('^', '_') + ' = ' + arg.name.replace('~', '_').replace('^', '_'))
        init_header = init_header.format(', '.join(arglist))
        super_stat = ''.join([self.indent * 2, 'super(', module.name, ', self).__init__()'])
        parlist = []
        pardiscs = []
        for par in module.pars:
            parlist.append(par.name.replace('~', '_').replace('^', '_'))
            pardisc = self.indent * 3 + par.name.replace('~', '_').replace('^', '_') + ': '
            if par.type is not None:
                pardisc = pardisc + 'type - ' + par.type
            if par.shape is not None:
                if isinstance(par.shape, str):
                    pardisc = pardisc + '; shape - ' + par.shape.replace('~', '_').replace('^', '_')
                else:
                    pardisc = pardisc + '; shape - <' + ', '.join([sh.replace('~', '_').replace('^', '_') for sh in par.shape]) + '>'
            if par.default is not None:
                pardisc = pardisc + '; default - ' + par.default
                parlist[-1] = parlist[-1] + '=' + par.default
            pardiscs.append(pardisc)
        forward_header = forward_header.format(', '.join(parlist))
        retdiscs = []
        for retgroup in module.retvals:
            for ret in retgroup:
                retdisc = self.indent * 3 + ret.name.replace('~', '_').replace('^', '_') + ': '
                if ret.type is not None:
                    retdisc = retdisc + 'type - ' + ret.type
                if ret.shape is not None:
                    if isinstance(ret.shape, str):
                        retdisc = retdisc + '; shape - ' + ret.shape.replace('~', '_').replace('^', '_')
                    else:
                        retdisc = retdisc + '; shape - <' + ', '.join([sh.replace('~', '_').replace('^', '_') for sh in ret.shape]) + '>'
                retdiscs.append(retdisc)
            retdiscs.append(self.indent * 3 + 'Or ---')
        if len(retdiscs):
            retdiscs.pop()
        long_str_sep = self.indent * 2 + '"""'
        args_header = self.indent * 2 + 'Arguments:'
        pars_header = self.indent * 2 + 'Parameters:'
        rets_header = self.indent * 2 + 'Retvals:'
        self.forward_body = []
        indent = 2
        for stat in module.body:
            self.generate_each(stat, indent)
        if len(module.args) == 0:
            retval = [class_header, forward_header, long_str_sep, pars_header, *pardiscs, rets_header, *retdiscs, long_str_sep, *self.forward_body]
        else:
            retval = [class_header, init_header, long_str_sep, args_header, *argdiscs, long_str_sep, super_stat, *list(self.members.values()), ' ',\
                  forward_header, long_str_sep, pars_header, *pardiscs, rets_header, *retdiscs, long_str_sep, *self.forward_body]
        return '\n'.join(retval)
    def _gen_expr(self, expr):
        if isinstance(expr, ModuleRef):
            return self._gen_moduleref(expr)
        elif isinstance(expr, Identifier):
            expr = expr.name.replace('~', '_').replace('^', '_')
            if expr in self.members:
                return 'self.' + expr
            else:
                return expr
        else:
            return str(expr)
    def _gen_precedence(self, highop, lowops, moduleref):
        retval = []
        if isinstance(moduleref.pars[0], ModuleRef) and moduleref.pars[0].ref in lowops:
            retval.append('(')
            retval.append(self._gen_expr(moduleref.pars[0]))
            retval.append(')')
        else:
            retval.append(self._gen_expr(moduleref.pars[0]))
        retval.append(f' {highop} ')
        if isinstance(moduleref.pars[1], ModuleRef) and moduleref.pars[1].ref in lowops:
            retval.append('(')
            retval.append(self._gen_expr(moduleref.pars[1]))
            retval.append(')')
        else:
            retval.append(self._gen_expr(moduleref.pars[1]))
        return ''.join(retval)
    def _gen_builtins(self, moduleref):
        return self.builtin_map[moduleref.ref](moduleref)
    def _gen_consts(self, moduleref):
        retval = ['torch.', moduleref.ref]
        if len(moduleref.pars) == 0:
            retval.append('([')
            if isinstance(moduleref.args[-1], str) and moduleref.args[-1] in {'"float"', '"long"', '"bool"', "'float'", "'long'", "'bool'"}:
                for arg in moduleref.args[:-1]:
                    retval.extend([self._gen_expr(arg), ','])
                retval[-1] = '], '
                retval.extend(['dtype=torch.', moduleref.args[-1][1:-1], ').cuda()'])
            else:
                for arg in moduleref.args:
                    retval.extend([self._gen_expr(arg), ','])
                retval[-1] = ']).cuda()'
        else:
            retval.extend(['_like(', self._gen_expr(moduleref.pars[0]), ').cuda()'])
        return ''.join(retval)    
    def _gen_sort(self, moduleref):
        retval = ['torch.sort(', self._gen_expr(moduleref.pars[0])]
        if len(moduleref.args) >= 1:
            retval.extend([', dim=', self._gen_expr(moduleref.args[0])])
            if len(moduleref.args) == 2:
                retval.extend([', descending=', self._gen_expr(moduleref.args[1])])
        return ''.join(retval + [')'])
    def _gen_index(self, moduleref):
        retval = []
        if isinstance(moduleref.pars[0], ModuleRef) and moduleref.pars[0].ref in {'+', '-', '*', '/', '//', '&&', '||', '!', '&', '|', '~', '#'}:
            retval.extend(['(', self._gen_expr(moduleref.pars[0]), ')'])
        else:
            retval.append(self._gen_expr(moduleref.pars[0]))
        retval.append('[')
        for par in moduleref.pars[1:]:
            if isinstance(par, IndexSlice):
                if par.low != ':':
                    retval.append(self._gen_expr(par.low))
                retval.append(':')
                if par.high != ':':
                    retval.append(self._gen_expr(par.high))
                if par.step is not None:
                    retval.append(':')
                    if par.step != ':':
                        retval.append(self._gen_expr(par.step))
                retval.append(',')
            elif par == ':':
                retval.extend([':', ','])
            else:
                retval.extend([self._gen_expr(par), ', '])
        retval[-1] = ']'
        return ''.join(retval)
    def _gen_shape(self, moduleref):
        retval = []
        if isinstance(moduleref.pars[0], ModuleRef) and moduleref.pars[0].ref in {'+', '-', '*', '/', '//', '&&', '||', '!', '&', '|', '~', '#'}:
            retval.extend(['(', self._gen_expr(moduleref.pars[0]), ')'])
        else:
            retval.append(self._gen_expr(moduleref.pars[0]))
        retval.extend(['.shape[', self._gen_expr(moduleref.args[0]), ']'])
        return ''.join(retval)
    def _gen_range(self, moduleref):
        if isinstance(moduleref.pars[0], int) and moduleref.pars[0] == 0 and isinstance(moduleref.pars[2], int) and moduleref.pars[2] == 1:
            return ''.join(['range(', self._gen_expr(moduleref.pars[1]), ')'])
        elif isinstance(moduleref.pars[2], int) and moduleref.pars[2] == 1:
            return ''.join(['range(', self._gen_expr(moduleref.pars[0]), ', ', self._gen_expr(moduleref.pars[1]), ')'])
        else:
            return ''.join(['range(', self._gen_expr(moduleref.pars[0]), ', ', self._gen_expr(moduleref.pars[1]), ', ', self._gen_expr(moduleref.pars[2]), ')'])
    def _gen_arange(self, moduleref):
        retval = ['torch.arange(']
        for par in moduleref.pars:
            retval.extend([self._gen_expr(par), ', '])
        retval[-1] = ')'
        return ''.join(retval)
    def _gen_scaler(self, moduleref):
        if 'math' not in self.python_modules:
            self.python_modules['math'] = 'import math'
        retval = ['math.', moduleref.ref[1:], '(', self._gen_expr(moduleref.pars[0]), ')']
        return ''.join(retval)
    # def _gen_linear(self, moduleref):
    #     if moduleref.alias is None:
    #         moduname = moduleref.ref
    #     elif moduleref.alias[0] == '@':
    #         moduname = moduleref.alias[1:]
    #     else:
    #         moduname = moduleref.ref + '_' + moduleref.alias
    #     if moduname not in self.members:
    #         init = [self.indent * 2, 'self.', moduname, ' = ', 'nn.Linear(']
    #         for arg in moduleref.args:
    #             argexpr = self._gen_expr(arg)
    #             if argexpr == 'True':
    #                 continue
    #             init.extend([argexpr, ', '])
    #         init[-1] = ')'
    #         self.members[moduname] = ''.join(init)
    #     forward = ['self.', moduname, '(', self._gen_expr(moduleref.pars[0]), ')']
    #     return ''.join(forward)
    def _gen_nns(self, moduleref):
        if moduleref.alias is None:
            moduname = moduleref.ref
        elif moduleref.alias[0] == '@':
            moduname = moduleref.alias[1:]
        else:
            moduname = moduleref.ref + '_' + moduleref.alias
        if moduname not in self.members:
            init = [self.indent * 2, 'self.', moduname, ' = ', 'nn.', self.nn_map[moduleref.ref], '(']
            modu = builtins[moduleref.ref]
            for i, arg in enumerate(moduleref.args):
                if arg == ':':
                    init.extend([modu.args[i].default])
                else:
                    init.extend([self._gen_expr(arg), ', '])
            init[-1] = ')'
            self.members[moduname] = ''.join(init)
        forward = ['self.', moduname, '(']
        for par in moduleref.pars:
            forward.extend([self._gen_expr(par), ', '])
        forward[-1] = ')'
        return ''.join(forward)
    def _gen_reduce(self, moduleref):
        retval = ['torch.', moduleref.ref, '(', self._gen_expr(moduleref.pars[0]), ', ', 'dim=']
        if len(moduleref.args) == 0:
            retval.append('-1)')
        else:
            retval.extend([self._gen_expr(moduleref.args[0]), ')'])
        return ''.join(retval)
    def _gen_unary(self, moduleref):
        retval = ['torch.', moduleref.ref, '(', self._gen_expr(moduleref.pars[0]), ')']
        return ''.join(retval)
    def _gen_comp(self, moduleref):
        return ' '.join([self._gen_expr(moduleref.pars[0]), moduleref.ref, self._gen_expr(moduleref.pars[1])])
    def _gen_arc(self, moduleref):
        retval = ['torch.a', moduleref.ref[3:], '(']
        for par in moduleref.pars:
            retval.extend([self._gen_expr(par), ', '])
        retval[-1] = ')'
        return ''.join(retval)
    def _gen_mul(self, moduleref):
        return self._gen_precedence(moduleref.ref, {'+', '-'}, moduleref)
    def _gen_sub(self, moduleref):
        return self._gen_precedence(moduleref.ref, {'+', '-'}, moduleref)
    def _gen_div(self, moduleref):
        return self._gen_precedence(moduleref.ref, {'+', '-', '*', '/', '//'}, moduleref)
    def _gen_add(self, moduleref):
        return ''.join([self._gen_expr(moduleref.pars[0]), ' + ', self._gen_expr(moduleref.pars[1])])
    def _gen_transpose(self, moduleref):
        retval = ['torch.transpose(', self._gen_expr(moduleref.pars[0]), ', ', self._gen_expr(moduleref.args[0]), ', ', self._gen_expr(moduleref.args[1]), ')']
        return ''.join(retval)
    def _gen_neg(self, moduleref):
        return ''.join(['-', self._gen_expr(moduleref.pars[0])])
    def _gen_and(self, moduleref):
        return self._gen_precedence(moduleref.ref, {'or'}, moduleref)
    def _gen_or(self, moduleref):
        return ''.join([self._gen_expr(moduleref.pars[0]), ' or ', self._gen_expr(moduleref.pars[1])])
    def _gen_not(self, moduleref):
        return ''.join(['not ', self._gen_expr(moduleref.pars[0])])
    def _gen_biand(self, moduleref):
        return self._gen_precedence('&', {'bior'}, moduleref)
    def _gen_bior(self, moduleref):
        return ''.join([self._gen_expr(moduleref.pars[0]), ' | ', self._gen_expr(moduleref.pars[1])])
    def _gen_tand(self, moduleref):
            # elif moduleref.ref == '&&':
        return ''.join(['torch.logical_and(', self._gen_expr(moduleref.pars[0]), ', ', self._gen_expr(moduleref.pars[1]), ')'])
    def _gen_tor(self, moduleref):
            # elif moduleref.ref == '||':
        return ''.join(['torch.logical_or(', self._gen_expr(moduleref.pars[0]), ', ', self._gen_expr(moduleref.pars[1]), ')'])
    def _gen_tnot(self, moduleref):
            # elif moduleref.ref == '!':
        return ''.join(['torch.logical_not(', self._gen_expr(moduleref.pars[0]), ')'])
    def _gen_tbiand(self, moduleref):
            # elif moduleref.ref == '&':
        return ''.join(['torch.bitwise_and(', self._gen_expr(moduleref.pars[0]), ', ', self._gen_expr(moduleref.pars[1]), ')'])
    def _gen_tbior(self, moduleref):
            # elif moduleref.ref == '|':
        return ''.join(['torch.bitwise_or(', self._gen_expr(moduleref.pars[0]), ', ', self._gen_expr(moduleref.pars[1]), ')'])
    def _gen_tbnot(self, moduleref):
            # elif moduleref.ref == '~':
        return ''.join(['torch.bitwise_not(', self._gen_expr(moduleref.pars[0]), ')'])
    def _gen_mm(self, moduleref):
            # elif moduleref.ref == 'matmul':
        return ''.join(['torch.mm(', self._gen_expr(moduleref.pars[0]), ', ', self._gen_expr(moduleref.pars[1]), ')'])
    def _gen_reshape(self, moduleref):
            # elif moduleref.ref == 'reshape':
        retval = []
        if isinstance(moduleref.pars[0], ModuleRef) and moduleref.pars[0].ref in {'+', '-', '*', '/', '//', '&&', '||', '!', '&', '|', '~', '#'}:
            retval.extend(['(', self._gen_expr(moduleref.pars[0]), ')'])
        else:
            retval.append(self._gen_expr(moduleref.pars[0]))
        retval.append('.view(')
        for arg in moduleref.args:
            if arg == ':':
                retval.extend(['-1', ', '])
            else:
                retval.extend([self._gen_expr(arg), ', '])
        if retval[-1][-1] != '(':
            retval[-1] = ')'
        retval = ''.join(retval)
        return retval
    def _gen_nonlinear(self, moduleref):
            # elif moduleref.ref in {'sigmoid', 'tanh', 'softmax', 'softmin', 'relu', 'leakyrelu'}:
        if 'F' not in self.python_modules:
            self.python_modules['F'] = 'from torch.nn import functional as F'
        return ''.join(['F.', moduleref.ref, '(', self._gen_expr(moduleref.pars[0]), ')'])
    def _gen_dropout(self, moduleref):
            # elif moduleref.ref == 'dropout':
        retval = ['F.dropout(', self._gen_expr(moduleref.pars[0]), ', ', self._gen_expr(moduleref.args[0]),\
                  ', ', 'training=self.training']
        if len(moduleref.args) == 2:
            retval.extend([', ', self._gen_expr(moduleref.args[1])])
        return ''.join(retval + [')'])
    def _gen_cat(self, moduleref):
        # elif moduleref.ref in {'concatenate', 'cat'}:
        retval = ['torch.cat([']
        for par in moduleref.pars:
            retval.extend([self._gen_expr(par), ', '])
        retval[-1] = '], '
        if len(moduleref.args) == 0:
            retval.append('dim=-1')
        else:
            retval.extend(['dim=', self._gen_expr(moduleref.args[0])])
        return ''.join(retval)
    def _gen_stack(self, moduleref):
        # elif moduleref.ref == 'stack':
        retval = ['torch.stack([']
        for par in moduleref.pars:
            retval.extend([self._gen_expr(par), ', '])
        retval[-1] = '], '
        if len(moduleref.args) == 0:
            retval.append('dim=-1')
        else:
            retval.extend(['dim=', self._gen_expr(moduleref.args[0])])
        return ''.join(retval)
    def _gen_squeeze(self, moduleref):
            # elif moduleref.ref in {'squeeze', 'unsqueeze'}:
        retval = ['torch.', moduleref.ref, '(']
        retval.append(self._gen_expr(moduleref.pars[0]))
        if len(moduleref.args) == 0:
            retval.append(', dim=-1')
        else:
            retval.extend([', dim=', self._gen_expr(moduleref.args[0])])
        retval.append(')')
        return ''.join(retval)
    def _gen_repeat(self, moduleref):
        retval = []
        if isinstance(moduleref.pars[0], ModuleRef) and moduleref.pars[0].ref in {'+', '-', '*', '/', '//', '&&', '||', '!', '&', '|', '~'}:
            retval.extend(['(', self._gen_expr(moduleref.pars[0]), ')'])
        else:
            retval.append(self._gen_expr(moduleref.pars[0]))
        retval.append('.repeat(')
        for arg in moduleref.args:
            if arg == ':':
                retval.extend(['1', ', '])
            else:
                retval.extend([self._gen_expr(arg), ', '])
        if retval[-1][-1] != '(':
            retval[-1] = ')'
        retval = ''.join(retval)
        return retval
    def _gen_expand(self, moduleref):
        retval = []
        if isinstance(moduleref.pars[0], ModuleRef) and moduleref.pars[0].ref in {'+', '-', '*', '/', '//', '&&', '||', '!', '&', '|', '~'}:
            retval.extend(['(', self._gen_expr(moduleref.pars[0]), ')'])
        else:
            retval.append(self._gen_expr(moduleref.pars[0]))
        retval.append('.expand(')
        for arg in moduleref.args:
            if arg == ':':
                retval.extend(['1', ', '])
            else:
                retval.extend([self._gen_expr(arg), ', '])
        if retval[-1][-1] != '(':
            retval[-1] = ')'
        retval = ''.join(retval)
        return retval
    def _gen_layernorm(self, moduleref):
        if moduleref.alias is None:
            moduname = moduleref.ref
        elif moduleref.alias[0] == '@':
            moduname = moduleref.alias[1:]
        else:
            moduname = moduleref.ref + '_' + moduleref.alias
        if moduname not in self.members:
            init = [self.indent * 2, 'self.', moduname, ' = ', 'nn.LayerNorm(']
            if len(moduleref.args) == 1:
                init.extend([self._gen_expr(moduleref.args[0]), ')'])
            else:
                init.append('[')
                for arg in moduleref.args:
                    argexpr = self._gen_expr(arg)
                    init.extend([argexpr, ', '])
                init[-1] = '])'
            self.members[moduname] = ''.join(init)
        forward = ['self.', moduname, '(', self._gen_expr(moduleref.pars[0]), ')']
        return ''.join(forward)        
    def _gen_moduleref(self, moduleref):
        if moduleref.alias is None:
            moduname = moduleref.ref
        elif moduleref.alias[0] == '@':
            moduname = moduleref.alias[1:]
        else:
            moduname = moduleref.ref + '_' + moduleref.alias
        if moduleref.ref in builtins:
            return self._gen_builtins(moduleref)
        else:
            # print(moduleref)
            if moduname not in self.members:
                init = [
                    self.indent * 2, 'self.', moduname, ' = ', moduleref.ref, '('
                ]
                modu = self.modus[moduleref.ref]
                for i, arg in enumerate(moduleref.args):
                    if isinstance(arg, str) and arg == ':':
                        init.extend([modu.args[i].default, ', '])
                    else:
                        init.extend([self._gen_expr(arg), ', '])
#                 while i < len(modu.args):
                    # tail omissions
                if init[-1] != '(':
                    init[-1] = ')'
                else:
                    init.append(')')
                self.members[moduname] = ''.join(init)
            forward = ['self.', moduname, '(']
            for par in moduleref.pars:
                forward.extend([self._gen_expr(par), ', '])
            forward[-1] = ')'
            return ''.join(forward)
    def generate_each(self, stat, indent):
        sentence = indent * self.indent
        if isinstance(stat, Statement):
            lhslist = []
            for var in stat.lhs:
                if isinstance(var, LocalVar):
                    if stat.static:
                        lhslist.append('self.' + var.name.replace('~', '_').replace('^', '_'))
                    else:
                        lhslist.append(var.name.replace('~', '_').replace('^', '_'))
                else:
                    lhslist.append(self._gen_expr(var))
            if isinstance(stat.rhs, ModuleRef) and stat.rhs.ref == 'lstm':
                lhslist[1] = '(' + lhslist[1]
                lhslist[2] = lhslist[2] + ')'
            eq = ' = '
            if isinstance(stat.rhs, ModuleRef):
                rhs = self._gen_moduleref(stat.rhs)
            elif isinstance(stat.rhs, Identifier):
                rhs = stat.rhs.name.replace('~', '_').replace('^', '_')
                if rhs in self.members:
                    rhs = 'self.' + rhs
            else:
                rhs = str(stat.rhs)
            if stat.static:
                lhs = ','.join(lhslist)
                sentence = sentence + lhs + eq + rhs
                self.members[lhslist[0][5:] if len(lhslist) else tuple(lhslist)] = sentence
            else:
                lhs = ','.join(lhslist)
                sentence = sentence + lhs + eq + rhs
                self.forward_body.append(sentence)
        elif isinstance(stat, IfElseBlock):
            for i, (cond, body) in enumerate(zip(stat.conds, stat.bodies)):
                if i == 0:
                    self.forward_body.append(self.indent * indent + 'if ' + self._gen_expr(cond) + ':')
                elif cond is None:
                    self.forward_body.append(self.indent * indent + 'else:')
                else:
                    self.forward_body.append(self.indent * indent + 'elif ' + self._gen_expr(cond) + ':')
                if len(body):
                    for substat in body:
                        self.generate_each(substat, indent + 1)
                else:
                    self.forward_body.append(self.indent * (indent + 1) + 'pass')
        elif isinstance(stat, ForRangeBlock):
            self.forward_body.append(self.indent * indent + 'for ' + self._gen_expr(stat.var) + ' in ' + self._gen_expr(stat.range) + ':')
            for substat in stat.body:
                self.generate_each(substat, indent + 1)
        elif isinstance(stat, Return):
            self.forward_body.append(self.indent * indent + 'return ' + ','.join([self._gen_expr(retval) for retval in stat.retvals]))
        else:
            raise NotImplementedError(str(type(stat)))