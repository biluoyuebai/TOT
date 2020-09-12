from nodes import Module, Parameter, Argument, Retval, VaArg, VaPar

builtins = {}

shape = Module('shape')
shape.args.append(Argument('index', 'long', '-1'))
shape.pars.append(Parameter('tensor'))
shape.retvals.append([Retval('shape', 'long', 'scaler')])
builtins['shape'] = shape    

size = Module('size')
size.args.append(Argument('index', 'long', '-1'))
size.pars.append(Parameter('tensor'))
size.retvals.append([Retval('size', 'long', 'scaler')])
builtins['size'] = size

arange = Module('range')
arange.pars.append(Parameter('low', 'long'))
arange.pars.append(Parameter('high', 'long'))
arange.pars.append(Parameter('step', 'long'))
arange.retvals.append([Retval('range_list')])
builtins['range'] = arange

arange = Module('arange')
arange.pars.append(Parameter('low', 'long'))
arange.pars.append(Parameter('high', 'long', default='0'))
arange.pars.append(Parameter('step', 'long', default='1'))
arange.retvals.append([Retval('range_tensor')])
builtins['arange'] = arange

linear = Module('linear')
linear.args.append(Argument('d_in', 'long'))
linear.args.append(Argument('d_out', 'long'))
linear.args.append(Argument('bias', 'bool', 'True'))
linear.pars.append(Parameter('x', 'float', ['...', 'd_in']))
linear.retvals.append([Retval('out', 'float', ['...', 'd_out'])])
builtins['linear'] = linear

for name in {'+', '-', '*', '/', '//', 'and', 'or', 'biand', 'bior', '&&', '||', '&', '|', '>', '<', '>=', '<=', '==', '!='}:
    modu = Module(name)
    modu.pars.append(Parameter('lhs'))
    modu.pars.append(Parameter('rhs'))
    modu.retvals.append([Retval('retval', 'same as lhs and rhs', 'shape as lhs and/or rhs')])
    builtins[name] = modu

for name in {'neg', 'not', 'binot', '!', '~', 'abs', 'sign', 'sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'log10', 'log2', 'arcsin', 'arccos', 'arctan'}:
    modu = Module(name)
    modu.pars.append(Parameter('x'))
    modu.retvals.append([Retval('retval', 'same as x', 'shape as x')])
    builtins[name] = modu

for name in {'sabs', 'ssign', 'ssqrt', 'ssin', 'scos', 'stan', 'slog', 'sexp', 'sceil', 'sfloor'}:
    modu = Module(name)
    modu.pars.append(Parameter('x'))
    modu.retvals.append([Retval('retval', 'same as x', 'shape as x')])
    builtins[name] = modu

dropout = Module('dropout')
dropout.args.append(Argument('prob', 'float'))
dropout.args.append(Argument('inplace', 'bool', 'False'))
dropout.pars.append(Parameter('x'))
dropout.retvals.append([Retval('retval', 'same as x', 'shape as x')])
builtins['dropout'] = dropout

for name in {'sigmoid', 'relu', 'tanh', 'leakyrelu', 'softmax', 'softmin'}:
    modu = Module(name)
    modu.args.append(Argument('inplace', 'bool', 'False'))
    modu.pars.append(Parameter('x', 'float'))
    modu.retvals.append([Retval('retval', 'float', 'shape as x')])
    builtins[name] = modu

transpose = Module('transpose')
transpose.args.append(Argument('dim1', 'long', "-1"))
transpose.args.append(Argument('dim2', 'long', "-2"))
transpose.pars.append(Parameter('x'))
transpose.retvals.append([Retval('retval', 'same as x', 'swap the dim size between `dim1` and `dim2`')])
builtins['transpose'] = transpose
T = Module('T')
T.args.append(Argument('dim1', 'long', "-1"))
T.args.append(Argument('dim2', 'long', "-2"))
T.pars.append(Parameter('x'))
T.retvals.append([Retval('retval', 'same as x', 'swap the dim size between dim1 and dim2')])
builtins['T'] = T

concatenate = Module('concatenate')
concatenate.args.append(Argument('dim', 'long', "-1"))
concatenate.pars.append(VaPar())
concatenate.retvals.append([Retval('retval', 'same as va_pars', 'sum the length of `dim`, remain the rest')])
builtins['concatenate'] = concatenate
cat = Module('cat')
cat.args.append(Argument('dim', 'long', "-1"))
cat.pars.append(VaPar())
cat.retvals.append([Retval('retval', 'same as va_pars', 'sum the length of `dim`, remain the rest')])
builtins['cat'] = cat

stack = Module('stack')
stack.args.append(Argument('dim', 'long', "0"))
stack.pars.append(VaPar())
stack.retvals.append([Retval('retval', 'va_pars', 'add an extra dim at `dim` with length of the sum of pars, remain the rest')])
builtins['stack'] = stack

reshape = Module('reshape')
reshape.args.append(VaArg('long', "-1"))
reshape.pars.append(Parameter('x'))
reshape.retvals.append([Retval('retval', 'same as x', 'shape as args')])
builtins['reshape'] = reshape

squeeze = Module('squeeze')
squeeze.args.append(Argument('dim', 'long', "-1"))
squeeze.pars.append(Parameter('x'))
squeeze.retvals.append([Retval('retval', 'same as x', 'shape - delete 1 dimension at `dim`, remain the rest')])
builtins['squeeze'] = squeeze

unsqueeze = Module('unsqueeze')
unsqueeze.args.append(Argument('dim', 'long', "-1"))
unsqueeze.pars.append(Parameter('x'))
unsqueeze.retvals.append([Retval('retval', 'same as x', 'shape - add 1 dimension at `dim`, remain the rest')])
builtins['unsqueeze'] = unsqueeze

where = Module('where')
where.pars.append(Parameter('cond', 'bool'))
where.pars.append(Parameter('if_'))
where.pars.append(Parameter('else_'))
where.retvals.append([Retval('retval', 'same as if_ and else_', "shape as cond, if_ and else_")])
builtins['where'] = where

clamp = Module('clamp')
clamp.args.append(Argument('low', default="0"))
clamp.args.append(Argument('high', default="0"))
clamp.pars.append(Parameter('x'))
clamp.retvals.append([Retval('retval', 'same as x', 'shape as x')])
builtins['clamp'] = clamp

pow = Module('pow')
pow.pars.append(Parameter('x'))
pow.pars.append(Parameter('y'))
pow.retvals.append([Retval('retval', 'float', 'shape as x or y')])
builtins['pow'] = pow

for name in ['sum', 'prod', 'max', 'min', 'argmax', 'argmin', 'mean', 'median', 'std', 'var']:
    modu = Module(name)
    modu.args.append(Argument('dim', 'long', "-1"))
    modu.pars.append(Parameter('x'))
    modu.retvals.append([Retval('retval', 'same as x', 'shape - remove the dim at `dim`, remain the rest')])
    builtins[name] = modu

sort = Module('sort')
sort.args.append(Argument('dim', 'long', "-1"))
sort.args.append(Argument('descending', 'bool', "False"))
sort.pars.append(Parameter('x'))
sort.retvals.append([Retval('sorted', 'same as x', 'shape as x'), Retval('indices', 'long', 'shape as x')])
builtins['sort'] = sort

repeat = Module('repeat')
repeat.args.append(VaArg('long', '1'))
repeat.pars.append(Parameter('x'))
repeat.retvals.append([Retval('retval', 'same as x', 'shape of expanded x')])
builtins['repeat'] = repeat

expand = Module('expand')
expand.args.append(VaArg('long', '1'))
expand.pars.append(Parameter('x'))
expand.retvals.append([Retval('retval', 'same as x', 'shape of expanded x')])
builtins['expand'] = expand

matmul = Module('matmul')
matmul.pars.append(Parameter('lhs'))
matmul.pars.append(Parameter('rhs'))
matmul.retvals.append([Retval('retval', 'same as lhs and rhs', 'the result of matrix multiplication')])
builtins['matmul'] = matmul

for name in ['conv1d', 'conv2d', 'conv3d']:
    modu = Module(name)
    modu.args.append(Argument('in_channels', 'long'))
    modu.args.append(Argument('out_channels', 'long'))
    modu.args.append(Argument('kernel_size', 'long or tuple[long]'))
    modu.args.append(Argument('stride', 'long or tuple[long]', "1"))
    modu.args.append(Argument('padding', 'long or tuple[long]', "0"))
    modu.args.append(Argument('dilation', 'long or tuple[long]', "0"))
    modu.args.append(Argument('groups', 'long', "1"))
    modu.args.append(Argument('bias', 'bool', "True"))
    modu.pars.append(Parameter('x', shape=['...', 'C_in', 'L_in']))
    modu.retvals.append([Retval('retval', 'same as x', ['...', 'C_out', 'L_out'])])
    builtins[name] = modu

for name in ['maxpool1d', 'maxpool2d', 'maxpool3d']:
    modu = Module(name)
    modu.args.append(Argument('kernel_size', 'long or tuple[long]'))
    modu.args.append(Argument('stride', 'long or tuple[long] or None', 'None'))
    modu.args.append(Argument('padding', 'long or tuple[long]', '0'))
    modu.args.append(Argument('dilation', 'long or tuple[long]', '1'))
    modu.pars.append(Parameter('x', shape=["...","C","H_in", "W_in"]))
    modu.retvals.append([Retval('retval', 'same as x', ['...', 'C', 'H_out', 'W_out'])])
    builtins[name] = modu

for name in ['avgpool1d', 'avgpool2d', 'avgpool3d']:
    modu = Module(name)
    modu.args.append(Argument('kernel_size', 'long or tuple[long]'))
    modu.args.append(Argument('stride', 'long or tuple[long] or None', 'None'))
    modu.args.append(Argument('padding', 'long or tuple[long]', '0'))
    modu.pars.append(Parameter('x', shape=["...","C","H_in", "W_in"]))
    modu.retvals.append([Retval('retval', 'same as x', ['...', 'C', 'H_out', 'W_out'])])
    builtins[name] = modu

for i, name in enumerate(['batchnorm1d', 'batchnorm2d', 'batchnorm3d']):
    modu = Module(name)
    modu.args.append(Argument('num_features', 'long'))
    modu.pars.append(Parameter('x', shape=['...', 'C'] + ['L'] * (i + 1)))
    modu.retvals.append([Retval('retval', 'same as x', ['...', 'C', '...'])])
    builtins[name] = modu

layernorm = Module('layernorm')
layernorm.args.append(VaArg('long or list[long]'))
layernorm.pars.append(Parameter('x', shape=['...']))
layernorm.retvals.append([Retval('retval', 'same as x', 'shape as x')])
builtins['layernorm'] = layernorm

rnn = Module('rnn')
rnn.args.append(Argument('input_size', 'long'))
rnn.args.append(Argument('hidden_size', 'long'))
rnn.args.append(Argument('num_layers', 'long', '1'))
rnn.args.append(Argument('nonlinearity', 'str', r'"tanh"'))
rnn.args.append(Argument('bias', 'bool', 'True'))
rnn.args.append(Argument('batch_first', 'bool', 'False'))
rnn.args.append(Argument('dropout', 'float', '0.0'))
rnn.args.append(Argument('bidirectional', 'bool', 'False'))
rnn.pars.append(Parameter('input', 'float', shape=['seq_len', 'batch', 'input_size']))
rnn.pars.append(Parameter('h_0', 'float', shape=['num_layers * num_directions', 'batch', 'hidden_size'], default='None'))
rnn.retvals.append([Retval('output', 'float', ['seq_len', 'batch', 'num_directions * hidden_size']), Retval('hidden_n', 'float', ['num_layers * num_directions', 'batch', 'hidden_size'])])
builtins['rnn'] = rnn

rnn = Module('lstm')
rnn.args.append(Argument('input_size', 'long'))
rnn.args.append(Argument('hidden_size', 'long'))
rnn.args.append(Argument('num_layers', 'long', '1'))
rnn.args.append(Argument('nonlinearity', 'str', r'"tanh"'))
rnn.args.append(Argument('bias', 'bool', 'True'))
rnn.args.append(Argument('batch_first', 'bool', 'False'))
rnn.args.append(Argument('dropout', 'float', '0.0'))
rnn.args.append(Argument('bidirectional', 'bool', 'False'))
rnn.pars.append(Parameter('input', 'float', shape=['seq_len', 'batch', 'input_size']))
rnn.pars.append(Parameter('h_0', 'float', shape=['num_layers * num_directions', 'batch', 'hidden_size'], default='None'))
rnn.pars.append(Parameter('c_0', 'float', shape=['num_layers * num_directions', 'batch', 'hidden_size'], default='None'))
rnn.retvals.append([Retval('output', 'float', ['seq_len', 'batch', 'num_directions * hidden_size']), Retval('hidden_n', 'float', ['num_layers * num_directions', 'batch', 'hidden_size']), Retval('cell_n', 'float', ['num_layers * num_directions', 'batch', 'hidden_size'])])
builtins['lstm'] = rnn

embedding = Module('embedding')
embedding.args.append(Argument('num_embeddings', 'long'))
embedding.args.append(Argument('embedding_dim', 'long'))
embedding.pars.append(Parameter('x', 'long'))
embedding.retvals.append([Retval('retval', 'float', ['...'])])
builtins['embedding'] = embedding

index = Module('[]')
index.args.append(VaArg(type='long', default=':'))
index.pars.append(Parameter('var'))
index.retvals.append([Retval('retval', 'float', ['...'])])
builtins['[]'] = index

for name in {'zeros', 'ones', 'empty', 'randn'}:
    modu = Module(name)
    modu.args.append(VaArg(type='long or str'))
    modu.pars.append(Parameter('like', default='None'))
    modu.retvals.append([Retval(name, shape='as args')])
    builtins[name] = modu

if __name__ == '__main__':
    for builtin in builtins.values():
        print(builtin)