# Transformer-如何设计和构建高效的时序模型
在自然语言处理(NLP)领域，RNN一直是被最广泛使用的深度机器学习模型，近年来CNN也逐渐被用于进行。。。然而这两类模型都有一些难以克服的问题，Transformer就是为了解决这些问题的新型模型，并取得了非常好的效果，大有取代RNN在NLP领域的统治地位的趋势，本文我们就来一步步的分析和理解这个优秀的seq2seq模型。

## 序列到序列问题（seq2seq）
seq2seq问题是使用机器学习（特别是深度学习）解决的一类常见问题，例如机器翻译，语态分析，摘要生成等自然语言处理问题（NLP），还包括_______。 这类问题的最大特点是输入（或输出）以序列的形式出现，序列的长度可变，任务通常要求分析整个序列才能产生输出————————。
### RNN
处理seq2seq问题的传统方法是使用RNN模型，RNN能够保存状态，它将输入分为多步，依靠每步输入和上一步的状态更新当前的状态（和输出），通过重复这种步骤在读入所有序列元素后从模型结构上来说特别适合序列到序列问题。问题有三点
1. 长序列的训练很困难
2. 只能顺序执行，训练速度很慢
3. 固定的存储不适合长序列
### CNN
CNN可以同时处理序列中的所有元素，但是由于卷及运算的视域有限，一次卷积操作只能处理有限的元素，对于较长的序列无法处理。解决办法是通过叠加多层卷积操作来逐渐增加视域，但这样会不可避免的导致信息丢失，并且仍没有完全解决长序列输入的处理问题，————————而且增加了模型的复杂度，使运算变慢，这和初衷不符。
### Attention机制
总结上述两种模型的处理方式，我们发现对于长序列的输入，无论是在预测准确度还是训练速度都有不足，有没有一种方法能从根本上解决这些问题，让我们一次性的看到全部输入（无论序列有多长），并且能根据这些输入信息分析他们之间的关联关系呢？答案就是attention机制，
图


## Transformer模型
基于attention机制
- 解决long memory problem
- 实现了部分并行运算，极大缩短了训练时间
- 提高了准确率
- bonus：为BERT打下了基础
### Encoder-Decoder architecture
Transformer的创新主要有以下几点
- Model architecture
	- 最大创新是只使用普通神经网络来实现seq2seq task，避免RNN和CNN的问题，在训练速度和准确率方面取得了双赢
	- RNN CNN free - help to speed up training
	- Stacking of encoder/decoder
- Multiple Attention Head
different random initial weights matrix may lead to different representation subspace, thus give transformer ability to understand different meaning of a word
- Position encoding

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
- why not positional index? extrapolate training samples
### point-wise FFN
### Mask
## Transformer实现
### layer normalization
### residual connection
- Help gradient propagated back through stacked decoders and encoders
- Residuals carry positional information to higher layers, among other information.
### warn-up learning rate
### regularization
- dropout
- layer normalization

## Resources
[Attention is all you need review]([https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html](https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html))
[The transformer - Attention is all you need]([https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XTEl6ugzZPY](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XTEl6ugzZPY))
<!--stackedit_data:
eyJoaXN0b3J5IjpbMzkzMDg1Mjg4LC0xMzE1Nzc0NzkwLDM1Mz
M5NDE4NSwtMTUxMTIxMzEyLDM2ODYwNjIyMCwyOTQyNjIzNjcs
MjA4MDgyMjA0MiwzNDM1MDcyMjUsMjkwMjQ3NTYyLDU3NzkzMj
AyOCwxMzIwOTc2NjE5LDE4MTg4OTg5MDYsLTEyNDUzMjY5NjAs
MTI5NjIzMzQ0NiwyMDM5NDg1NzI5LDU3Mjc5MzAxOCwtMTA4OD
c1NDUzMiw1MzIwMTMzNjEsMjcwMzMzNTM3LDE5MTMyNjM5OThd
fQ==
-->