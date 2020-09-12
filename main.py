import argparse
from parser import file_to_file
from pytorch_generator import PyTorchGenerator

if __name__ == "__main__":
    print("Tensor Operations Tool - TOT")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='path to the file to transform')
    parser.add_argument('--output', '-o', type=str, help='path to place output file')
    args = parser.parse_args()
    file_to_file(args.input, args.output, PyTorchGenerator())

"""
class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores += mask
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)
"""

# pc = ParserContext()
# # pc.init_modu('test')
# # pc.add_arg('d_in', 'long')
# # pc.add_arg('d_hidden', 'long')
# # pc.add_arg('d_out', 'long')
# # pc.add_par('x', 'float', ['...', 'd_in'])
# # pc.add_retval('x', 'float', ['...', 'd_out'])
# # pc.launch_parser()
# # pc.parse('x = linear{0}[d_in, d_hidden](x)')
# # pc.parse('x = linear{1}[d_hidden, d_hidden](x)')
# # pc.parse('x = linear{2}[d_hidden, d_out](x)')
# # pc.parse('return x')
# pc.init_modu('ScaledDotProductAttention')
# pc.add_par('query', 'float', ['...', 'l_q', 'd_q'])
# pc.add_par('key', 'float', ['...', 'l_k', 'd_k'])
# pc.add_par('value', 'float', ['...', 'l_k', 'd_k'])
# pc.add_retval('value_out', 'float', ['...', 'l_q', 'd_k'], 'att~scores', 'float', ['...', 'l_q', 'l_k'])
# pc.launch_parser()
# pc.parse('dk = shape[-1](query)')
# pc.parse('scores = matmul(query, T[-1,-2](key)) / ssqrt(dk)')
# pc.parse('scores = softmax[-1](scores)')
# pc.parse('return matmul(scores, value), scores')
# generator = PyTorchGenerator()
# print(pc.generate(generator))
# print('###################')
# print()

# pc.save_modu()
# pc.kill_parser()
# pc.init_modu('ScaledDotProductRelationAttention')
# pc.add_arg('head~num', 'long')
# pc.add_par('query', 'float', ['batch~size * head~num', 'node~num', 'd_q'])
# pc.add_par('key', 'float', ['batch~size * head~num', 'node~num', 'd_k'])
# pc.add_par('value', 'float', ['batch~size * head~num', 'node~num', 'd_k'])
# pc.add_par('relation', 'float', ['batch~size', 'node~num', 'node~num', 'd_r'])
# pc.add_par('mask', 'float', ['batch~size * head~num', 'node~num', 'node~num'], 'None')
# pc.add_retval('value_out', 'float', ['batch~size', 'node~num', 'd_q'])
# pc.launch_parser()
# pc.parse('dk = shape[-1](query)')
# pc.parse('scores = matmul(query, T[-1,-2](key)) / ssqrt(dk)')
# pc.parse('relation = reshape[,shape[1](scores),shape[1](scores),dk](repeat[,head~num,,,](unsqueeze[1](relation)))')
# pc.parse('relative~scores = squeeze(matmul(unsqueeze[-2](query), T[-1,-2](relation)))')
# pc.parse('scores = scores + relative~scores')
# pc.parse('scores = reshape[,head~num,shape[1](scores),shape[1](scores)](scores)')
# pc.parse('scores = scores + mask')
# pc.parse('scores = reshape[,shape[2](scores),shape[2](scores)](scores)')
# pc.parse('attention = softmax[-1](scores)')
# pc.parse('self~att~val = matmul(attention, value)')
# pc.parse('attention = unsqueeze[-2](attention)')
# pc.parse('rel~att~val = matmul(attention, relation)')
# pc.parse('return self~att~val + rel~att~val')
# print(pc.generate(generator))