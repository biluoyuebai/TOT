# Tensor Operations Tool（TOT） $\alpha$版

## 1. module 整体框架

以下 尖括号 `<>` 内的内容为用户必填可替换内容，圆括号内的内容为用户可选填可替换内容`()`  
此外，用作分隔符的 `|` 根据其后是否选填而存在  
注意这里的尖括号和圆括号不是语法的一部分

>module `<module name>`  
>arg `<argname>` | `(argtype)` | `(argdefault)`  
>`(more args)`  
>par `<parname>` | `(partype)` | `(parshape)` | `(pardefault)`  
>`(more pars)`  
>ret `<retname>` | `(rettype)` | `(retshape)`   
>`(more rets)`  
>`<body> or <forward>`  
>`<module contents>`  
>moduleend

详解：  

1. 合法的`module name`参照合法的 python 变量名；即可用作 python 变量名的标识符均是合法的`module name`。
2. `arg`表示用于构建该模块`module`的静态参数，`par`表示输入该模块的数据，`ret`表示该模块的返回值。
3. 合法的`argname`、`parname`、`retname`的格式为`ident(~sub)(^sup)`，其中`ident`为合法的 python 变量名，`~sub`会转换为标识符的下标（对支持下标的后端，例如$\LaTeX$），`^sup`会转换为标识符的上标。
4. 合法的`argtype`、`partype`、`rettype`包括：`float`、`int`、`bool`、`LongTensor`、`FloatTensor`、`BoolTensor`、`ByteTensor`
5. 合法的`parshape`、`retshape`为以逗号`,`分割的各个维度的大小
6. 请务必按照`args`、`pars`、`rets`的顺序填写，允许缺失`args`，但不允许缺失`pars`和`rets`
7. 暂不支持多`rets`组，即不允许通过`if-else`分支按条件返回数量不等的`rets`，即一个`module`中只允许存在一条`return`
8. 注释格式：以 `#` 开头的行视作注释，不会被解析。（$\alpha$ - v0.0.2 新增）
9. 样例请参见`demo`中文档

## 2. 一般语句

### 2.1. 整体格式

> `<var0>`,`(var1)`,`(var2)`, ... = `<expr>`

例如

> `sorted, indices = att.softmax.sort`

详解：

1. 等号左端为`<expr>`表达式的返回值，必须是单个值，（除了切片索引外，$\alpha$ - v0.0.2 实现）不允许用一个函数调用的返回值承接（按：不允许`lhs() = rhs()`），程序会静态检查左端返回值的个数是否和`<expr>`实际返回值的个数匹配。
2. 合法的临时变量的格式和`argname`、`parname`、`retname`相同
3. 不允许左端变量和`args`重名（系统会静态识别）
4. `<expr>`的格式包含允许嵌套的一般函数、一元函数、下标索引、加减乘除法、括号、矩阵乘法、常量
5. 如果担心一行写不下，可以在行尾加一个`\`，类似 python，然后从下一行开始写。
6. 本语法格式和缩进无关，会自动剔除行首的空格，所以你可以按照个人缩进方式优化代码的视觉感受。

### 2.2. 一般函数

> `funcname{alias}[args](pars)`

例如

> `ScaledDotProductAttention{@attention}[head_num](q, k, v, mask)`  
> `linear{in}[d_in, d_hid](input)`  
> `sqrt(x)`  
> `repeat[,,,d~f*head_num](x)`

详解：

1. `funcname`为已经定义了的模块的名字，已经定义的模块包括：内建模块`builtins`（参见第三章`builtins`一节）、当前文档之前定义的模块、引入的第三方模块（未完工）
2. `{alias}`为模块的别名，可以缺省，如果缺省则默认以模块的名称作为模块实例的名称。以`@`开头的别称之后导引的名称直接作为模块实例的名称，否则将`funcname`和`alias`以`_`连接作为实例的名称。
3. `[args]`为模块的`args`静态参数，可以缺省，表示无静态参数，或者所有的`args`都使用该模块的默认参数。内部的`args`以逗号`,`分割。如果想使用默认参数，则在其位置上留空即可（如上例中的`repeat`，逗号之间为缺省部分，使用`repeat`的默认值`1`）。
4. `(pars)`为模块的`pars`输入参数，不允许缺省。若只有一个`par`也可以使用后节叙述的一元函数语法。
5. `args`和`pars`均为可嵌套的，即允许调用其他表达式生成之（如上例中的`repeat`）

### 2.3. 一元函数

> `var.funcname{alias}[args]`

例如

> `input.linear[d_in,d_out].softmax.dropout[prob]`

详解：

1. `funcname`、`alias`、`args` 的格式和一般函数完全相同
2. `var`可以是变量，也可以是嵌套的表达式，但必须符合运算符结合律，即`a * b.softmax`和`(a * b).softmax`是不一样的。
3. 此设计为了从左到右按顺序将网络更清晰地展示出来，而非 pytorch keras mxnet 等从内而外逐个调用。

### 2.4. 索引运算

> `var[ind0,ind1,,,indn]`

例如

> `array[,,,ind]`

详解：

1. 被省略的维度视作 numpy 和 pytorch 中的`:`。
2. ~~暂不支持切片索引，即`var[left:right:step]`~~（$\alpha$ - v0.0.2 实现）

### 2.5. 预定义运算符

算术计算运算符：

`-`：取相反数，第二等优先级（次于函数调用）  
`#`：矩阵乘法，第三等优先级  
`*`：标量乘法、张量数乘、或者张量的逐元素乘法，第四等优先级  
`/`：标量除法、张量数除、或者张量的逐元素除法，第四等优先级  
`//`：标量整除，第四等优先级  
`+`：标量加法、张量加法，第五等优先级  
`-`：标量减法、张量减法，第五等优先级  
`()`：括号，将优先级提到最高

布尔计算运算符：

`and`：标量与  
`or`：标量或  
`not`：标量否  
`biand`：标量按位与  
`bior`：标量按位或  
`binot`：标量按位否  
`&&`：布尔张量与  
`||`：布尔张量或  
`!`：布尔张量否  
`&`：布尔张量按位与  
`|`：布尔张量按位或  
`~`：布尔张量按位否

按：不允许直接将算术计算运算符和布尔计算运算符混用，也即不支持布尔值到算术值的隐式转换，除非分别用作函数的不同实参、或者通过比较运算符混用。布尔计算运算符的优先级和 python 相同。

比较运算符：

按：从略，所有 python 支持的比较运算符均支持。但不支持 python 中支持的链式比较语法。允许比较运算符和布尔计算运算符混用，如`(1 + 3) > thresh and sum(logits) < 1.0`（功能未测试）。

## 3. 语法块

按：此部分尚未进行有效性测试，文档暂时略写，想尝试请谨慎使用。注意和 python 不同的是，不需要冒号的使用。注意下述尖括号不是语法的一部分。注意所有语法块都以`end`结尾。

### 3.1. if-elif-else

> `if <condition>`  
> `	//....`  
> `elif <condition>`  
> `//....`  
> `else`  
> `//....`  
> `end`

### 3.2. for-range

> `for <var> in <[low, high, step]>`  
> `//...`  
> `end`

按：表示范围的`[low, high, step]`可以有多种变体；如果`[]`中只有一个元素，则将其视作`high`，如`for i in [n_steps]`；两个元素，则将其视作`low`和`high`；三个元素则为一般情形。

### 3.3. for-each

（功能未完成）

## 4. 内建模块

执行`python builtin_modules.py`会打印所有可用的内建模块的调用原型。

按：以`s`开头的一些 element-wise 运算（例如 abs -> sabs, sqrt -> ssqrt）表示标量运算，不加`s`开头则表示张量运算。va_arg 和 va_par 表示参数列表是可变数量的。

按2：部分内建函数尚不支持，例如 rnn 和 lstm。

## 5. 开发日志

$\alpha$ - v0.0.2  
时间：2020年9月12日  
增加功能：现在等号左边可以索引赋值了（见下例）  
					增加切片索引功能`var[,1::2] = x.sin`  
					增加 CNN 相关内建函数（conv123d、maxpool123d、avgpool123d、batchnorm123d）  
					增加内建函数 arange、zeros、ones、empty、randn、zeros_like、ones_like、empty_like、randn_like  
					增加注释语法

$\alpha$ - v0.0.1  
时间：2020年9月10日  
基础功能：包含模块整体结构、主要功能语句、少数语句块、多数内建函数

## 6. 两个示例

> module BottleNeck
> arg in_channels | int
> arg out_channels | int
> arg stride | int | 1
> par x | FloatTensor | batch_size, ... in_channels, in_height, in_width
> ret x | FloatTensor | batch_size, ..., out_channels, out_height, out_width
> body
> residual = x.conv2d[in_channels, out_channels, 1]\
>             .batchnorm2d[out_channels].relu[True]\
>             .conv2d[out_channels,out_channels, 3, stride, 1]\
>             .batchnorm2d[out_channels].relu[True]\
>             .conv2d[out_channels, out_channels * 4, 1]\
>             .batchnorm2d[out_channels * 4]
> shortcut = x.conv2d[in_channels, out_channels * 4, 1, stride]\
>             .batchnorm2d[out_channels * 4]
> return (residual + shortcut).relu[True]
> moduleend

```python
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Arguments:
            in_channels: type - int
            out_channels: type - int
            stride: type - int; default - 1
        """
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.batchnorm2d = nn.BatchNorm2d(self.out_channels * 4)
        self.conv2d = nn.Conv2d(self.out_channels, self.out_channels * 4, 1)
 
    def forward(self, x):
        """
        Parameters:
            x: type - FloatTensor; shape - <batch_size, ... in_channels, in_height, in_width>
        Retvals:
            x: type - FloatTensor; shape - <batch_size, ... in_channels, in_height, in_width>
        """
        residual = self.batchnorm2d(self.conv2d(F.relu(self.batchnorm2d(self.conv2d(F.relu(self.batchnorm2d(self.conv2d(x))))))))
        shortcut = self.batchnorm2d(self.conv2d(x))
        return F.relu(residual + shortcut)
```

> module MultiHeadAttention
> arg input_dim | long
> arg d~model | long
> arg head_num | long
> par query | FloatTensor
> par key | FloatTensor
> par value | FloatTensor
> par mask | FloatTensor | None
> ret value_out | FloatTensor
> body
> q = query.linear{q}[input_dim, d~model]
> k = key.linear{k}[input_dim, d~model]
> v = value.linear{v}[input_dim, d~model]
> batch_size = q.shape[0]
> d~f = q.shape[-1]
> sub_dim = d~f // head_num
> q = q.reshape[batch_size,,head_num,sub_dim].T[1,2].reshape[batch_size*head_num,,sub_dim]
> k = k.reshape[batch_size,,head_num,sub_dim].T[1,2].reshape[batch_size*head_num,,sub_dim]
> v = v.reshape[batch_size,,head_num,sub_dim].T[1,2].reshape[batch_size*head_num,,sub_dim]
> if mask != None
>     mask = mask.repeat[head_num,,]
> end
> value_out = ScaledDotProductAttention{@attention}(q, k, v, mask)
> value_out = value_out.reshape[batch_size,head_num,,d~f].T[1,2].reshape[batch_size,,d~f*head_num]
> return value_out.linear{o}[d~model, input_dim]
> moduleend

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, head_num):
        """
        Arguments:
            input_dim: type - long
            d_model: type - long
            head_num: type - long
        """
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.head_num = head_num
        self.linear_q = nn.Linear(self.input_dim, self.d_model)
        self.linear_k = nn.Linear(self.input_dim, self.d_model)
        self.linear_v = nn.Linear(self.input_dim, self.d_model)
        self.attention = ScaledDotProductAttention()
        self.linear_o = nn.Linear(self.d_model, self.input_dim)
 
    def forward(self, query, key, value, mask):
        """
        Parameters:
            query: type - FloatTensor; shape - <...>
            key: type - FloatTensor; shape - <...>
            value: type - FloatTensor; shape - <...>
            mask: type - FloatTensor; shape - <None>
        Retvals:
            value_out: type - FloatTensor; shape - <...>
        """
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        batch_size = q.shape[0]
        d_f = q.shape[-1]
        sub_dim = d_f // self.head_num
        q = torch.transpose(q.view(batch_size, -1, self.head_num, sub_dim), 1, 2).view(batch_size * self.head_num, -1, sub_dim)
        k = torch.transpose(k.view(batch_size, -1, self.head_num, sub_dim), 1, 2).view(batch_size * self.head_num, -1, sub_dim)
        v = torch.transpose(v.view(batch_size, -1, self.head_num, sub_dim), 1, 2).view(batch_size * self.head_num, -1, sub_dim)
        if mask != None:
            mask = mask.repeat(self.head_num, 1, 1)
        value_out = self.attention(q, k, v, mask)
        value_out = torch.transpose(value_out.view(batch_size, self.head_num, -1, d_f), 1, 2).view(batch_size, -1, d_f * self.head_num)
        return self.linear_o(value_out)
```





















