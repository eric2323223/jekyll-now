# Transformer-如何设计和构建高效的时序模型
在自然语言处理(NLP)领域，RNN一直是被最广泛使用的深度机器学习模型，近年来CNN也逐渐被用于进行。。。然而这两类模型都有一些难以克服的问题，Transformer就是为了解决这些问题的新型模型，并取得了非常好的效果，大有取代RNN在NLP领域的统治地位的趋势，本文我们就来一步步的分析和理解这个优秀的seq2seq模型。

## 序列到序列问题（seq2seq）
seq2seq问题是使用机器学习（特别是深度学习）解决的一类常见问题，例如机器翻译，语态分析，摘要生成等自然语言处理问题（NLP），还包括_______。 这类问题的最大特点是输入（或输出）以序列的形式出现，序列的长度可变，任务通常要求分析整个序列才能产生输出————————。
### RNN
处理seq2seq问题的传统方法是使用RNN模型，RNN能够保存状态，它将输入分为多步，依靠每步输入和上一步的状态更新当前的状态（和输出），通过重复这种步骤在读入所有序列元素后得到整个序列的内部表示（latent feature vector）。从模型结构上来说特别适合序列到序列问题。问题有三点
1. 长序列的训练很困难
2. 只能顺序执行，训练速度很慢
3. 固定的存储不适合长序列
### CNN
CNN可以同时处理序列中的所有元素，但是由于卷积运算的视域有限，一次卷积操作只能处理有限的元素，对于较长的序列无法处理。解决办法是通过叠加多层卷积操作来逐渐增加视域，但这样会不可避免的导致信息丢失，并且仍没有完全解决长序列输入的处理问题，————————而且增加了模型的复杂度，使运算变慢，这和初衷不符。
### Attention机制
总结上述两种模型的处理方式，我们发现对于长序列的输入，无论是在预测准确度还是训练速度都有不足，有没有一种方法能从根本上解决这些问题，让我们一次性的看到全部输入（无论序列有多长），并且能根据这些输入信息分析序列元素之间的关联关系呢？答案就是attention机制，
图


## Transformer模型
基于attention机制
- 解决long memory problem
- 实现了部分并行运算，极大缩短了训练时间
- 提高了准确率

### 模型架构
整体架构上看，transformer仍属于Encoder-Decoder架构，通过encoder将输入序列转换成内部表示，在通过不同decoder实现不同的预测功能。
Transformer的最大的创在于它使用attention和全连接网络来实现seq2seq task，避免使用RNN和CNN从而使得在训练速度和准确率上全面超越了已有的方法。具体来讲

#### 新问题
- 位置编码Positional encoding
由于transformer不使用RNN和CNN，仅仅计算不同元素之间的相似度，因此必须加入位置信息来保证transformer正确的理解输入序列。最简单的位置编码是直接使用元素的序号，但这种方式对输入序列的长度过于敏感，对相对位置关系的表达——————。 extrapolate training samples
Transformer中使用了sin/cos位置编码
	1. 计算方便
	2. 能够体现相对位置关系
	3. 可处理变长序列
#### 改进和增强 
- Multiple Attention Head(MHA)
different random initial weights matrix may lead to different representation subspace, thus give transformer ability to understand different meaning of a word
 3. self attention
 4.  stack of encoder/decoder layer
	- - 位置编码Positional encoding
由于transformer不使用RNN 和CNN free - help to speed up training
	- Stacking of encoder/decoder
	- self attention， encoding-decoding attention


- Attention 
	- Use of self-attention to improve accuracy
	- Assumption: the more similar the more it contribute
	- Essence of Attention mechanism: **Feature reconstruction** based on all other inputs
	- Mathematically: weighted average
	- can be used in different tasks (text, visual, voice ...)
	- 3 types of attention

- **multi-head attention** VS convolution on multiple channels
	- Convolution: Different linear transformations by relative position
	- MHA: a weighted average 
	- It is found empirically that multi-head attention works better than the usual “single-head” in the context of machine translation. And the intuition behind such an improvement is that “multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions”
### Why multiple layer of attention layers?
### Vector similarity
### Positional encoding
- why not positional index? 
### point-wise FFN
### Mask



## Transformer实现
### layer normalization
### residual connection
- Help gradient propagated back through stacked decoders and encoders
- Residuals carry positional information to higher layers, among other information.
### warn-up learning rate
### regularization
## Transformer的改进
Despite not having any explicit recurrency, implicitly the model is built as an autoregressive one. It implies that in order to generate an output (both while training or during inference), the model needs to compute previous outputs, which is extremely costly, for the whole net has to be run for every output. That’s the main idea to overcome in a recent paper by researchers at [_Salesforce Research_](https://einstein.ai/research/non-autoregressive-neural-machine-translation) and the University of Hong Kong, who tried to make the whole process parallelizable[23](https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html#fn:23). Their proposal is to compute _fertilities_ for every input word in the sequence, and use it instead of previous outputs in order to compute the current output. This is summarized in the figure below.
## 总结
- dropout
- layer normalization

## Resources
[Attention is all you need review]([https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html](https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html))
[The transformer - Attention is all you need]([https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XTEl6ugzZPY](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XTEl6ugzZPY))
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExNjE4NjU3MzMsLTE0Nzc0MDYyOTMsLT
ExMDkyMTkxMjEsNTc5NjAwNDMxLDU3NDE3Nzc4LDc4NzY4NTIz
NiwxNzMwODcwNjc0LC0xODE1MDcwNDU3LC0xNjA0NzQwOTk1LD
kwNzMxOTkzOCwyMTM2NzA2OTI0LC0xMzE1Nzc0NzkwLDM1MzM5
NDE4NSwtMTUxMTIxMzEyLDM2ODYwNjIyMCwyOTQyNjIzNjcsMj
A4MDgyMjA0MiwzNDM1MDcyMjUsMjkwMjQ3NTYyLDU3NzkzMjAy
OF19
-->