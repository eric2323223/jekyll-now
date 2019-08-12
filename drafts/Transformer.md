# Transformer-如何设计和构建高效的时序模型
在自然语言处理(NLP)领域，RNN一直是被最广泛使用的深度机器学习模型，近年来CNN也逐渐被用于进行。。。然而这两类模型都有一些难以克服的问题，Transformer就是为了解决这些问题的新型模型，并取得了非常好的效果，大有取代RNN在NLP领域的统治地位的趋势，本文我们就来一步步的分析如何设计和实现transformer

## 序列到序列问题（seq2seq）
seq2seq问题是使用机器学习（特别是深度学习）解决的一类常见问题，例如机器翻译，语态分析，摘要生成等自然语言处理问题（NLP），还包括_______。 这类问题的最大特点是输入随着时间依次到来，输出————————。处理seq2seq的传统方法是使用RNN模型，RNN能够保存状态，它将输入分为多步，依靠每步输入和上一步的状态决定下一步的状态（和输出），从模型结构上来说特别适合序列到序列问题。问题yousuandia

## RNN和CNN
为什么要引入新的模型？加速训练，提高准确性
### RNN
- 无法并行运算
- Long memory problem
### CNN
- CNN由于使用尺寸受限的卷积核（convolution kernel）扫描输入数据，同样面临着long memory problem，要使CNN能够一次扫描大量的输入（长句子）就需要叠加多层卷积运算来实现，这样做的代价是增加了模型的复杂度，使运算变慢，这和初衷不符。

## Theory and Model
RNN解决了哪些问题，如何替代它？

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
## Training tricks
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
eyJoaXN0b3J5IjpbLTEyMDAxMzYyNTgsMjAzOTQ4NTcyOSw1Nz
I3OTMwMTgsLTEwODg3NTQ1MzIsNTMyMDEzMzYxLDI3MDMzMzUz
NywxOTEzMjYzOTk4LC0yNTI1MTE2NSwxMTE3MDAyMSw5OTcyMD
MwMzYsLTc5NzUyNTU2NywtMzUwMDUzNzc3LC0xNDA5MDQwNDEz
LDEwMDU3OTA0NTksLTEyMjQ5ODY5NjgsMTA3MzYwODAzOSwtMT
Q4MjU0MzI1NCwxNDEwMjgyMTM2LC00NTkzMzEyNTgsLTU2NzY3
NTM1OF19
-->