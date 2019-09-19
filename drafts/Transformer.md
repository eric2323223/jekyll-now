# Transformer-如何设计和构建高效的时序模型
在自然语言处理(NLP)领域，RNN一直是被最广泛使用的深度机器学习模型，近年来CNN也逐渐被用于进行。。。然而这两类模型都有一些难以克服的问题，Transformer就是为了解决这些问题的新型模型，并取得了非常好的效果，大有取代RNN在NLP领域的统治地位的趋势，本文我们就来一步步的分析和理解这个优秀的seq2seq模型。

## 序列到序列问题（seq2seq）
seq2seq问题是使用机器学习（特别是深度学习）解决的一类常见问题，例如机器翻译，语态分析，摘要生成等自然语言处理问题（NLP），还包括_______。 这类问题的最大特点是输入（或输出）以序列的形式出现，序列的长度可变，任务通常要求分析整个序列才能产生输出————————。使用机器学习（深度学习）处理seq2seq任务，通常使用编码器-解码器（encoder-decoder）架构，编码器负责将输入序列转换为整个序列的内部表示（context vector），解码器则对这个内部表示进行解释。
![enter image description here](https://img-blog.csdn.net/20180627114128329?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hwdWxmYw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)传统上有两类模型：
- RNN
处理seq2seq问题的传统方法是使用RNN模型，RNN能够保存状态，它将输入分为多步，依靠每步输入和上一步的状态更新当前的状态（和输出），通过重复这种步骤在读入所有序列元素后得到整个序列的内部表示（latent feature vector）。![enter image description here](https://miro.medium.com/max/2658/1*Ismhi-muID5ooWf3ZIQFFg.png)
从模型结构上来说特别适合序列到序列问题。问题有三点
1. 长序列的训练很困难
2. 只能顺序执行，训练速度很慢
3. 固定的存储不适合长序列
- CNN
CNN可以同时处理序列中的所有元素，但是由于卷积运算的视域有限，一次卷积操作只能处理有限的元素，对于较长的序列无法处理。解决办法是通过叠加多层卷积操作来逐渐增加视域，但这样会不可避免的导致信息丢失，并且仍没有完全解决长序列输入的处理问题，————————而且增加了模型的复杂度，使运算变慢，这和初衷不符。

总结一下，上述两种模型对于长序列的处理都有缺陷。RNN需要一步一步的处理输入序列，CNN做出了一些改进但并不彻底。从根本上的解决这个问题需要能一次性的处理全部输入（无论序列有多长），并且能根据这些输入信息分析序列元素之间的关联关系。人们从自己快速浏览的方式获得了启发，当人们需要快速浏览的时候一般不会一字一句的阅读，而会直接跳到需要关注的的部分，这种根据需要在不同位置跳跃的阅读方式和注意力相关，因此这种新的序列处理方式被命名为注意力机制

Attention机制来自于人类视觉注意力机制。人们视觉在感知东西的时候一般不会是一个场景从到头看到尾每次全部都看，而往往是根据需求观察注意特定的一部分。而且当人们发现一个场景经常在某部分出现自己想观察的东西时，人们会进行学习在将来再出现类似场景时把注意力放到该部分上。
> In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention.

图
	- Use of self-attention to improve accuracy
	- Assumption: the more similar the more it contribute
	- Essence of Attention mechanism: **Feature reconstruction** based on all other inputs
	- Mathematically: weighted average
	- can be used in different tasks (text, visual, voice ...)
	- 3 types of attention
> Attention is a method for aggregating a set of vectors  vivi  into just one vector, often via a lookup vector  uu. Usually,  vivi  is either the inputs to the model or the hidden states of previous time-steps, or the hidden states one level down (in the case of stacked LSTMs).
> 
> The result is often called the context vector  cc, since it contains
> the  _context_  relevant to the current time-step.
> 
> This additional context vector  cc  is then fed into the RNN/LSTM as
> well (it can be simply concatenated with the original input).
> Therefore, the context can be used to help with prediction.
> 
> The simplest way to do this is to compute probability vector 
> p=softmax(VTu)p=softmax(VTu)  and  c=∑ipivic=∑ipiviwhere  VV  is the
> concatenation of all previous  vivi. A common lookup vector  uu  is
> the current hidden state  htht.
> 
> There are many variations on this, and you can make things as
> complicated as you want. For example, instead using  vTiuviTu  as the
> logits, one may choose  f(vi,u)f(vi,u)  instead, where  ff  is an
> arbitrary neural network.
> 
> A common attention mechanism for sequence-to-sequence models uses 
> p=softmax(qTtanh(W1vi+W2ht))p=softmax(qTtanh⁡(W1vi+W2ht)), where  vv 
> are the hidden states of the encoder, and  htht  is the current hidden
> state of the decoder.  qq  and both  WWs are parameters.
> 
> Some papers which show off different variations on the attention idea:
> 
> [Pointer Networks](https://arxiv.org/abs/1506.03134)  use attention to
> reference inputs in order to solve combinatorial optimization
> problems.
> 
> [Recurrent Entity Networks](https://arxiv.org/abs/1612.03969) 
> maintain separate memory states for different entities
> (people/objects) while reading text, and update the correct memory
> state usingf ttention attention.
> 
> [Transformer](https://arxiv.org/pdf/1706.03762.pdf)  models also make
> extensive use of attention. Their formulation of attention is slightly
> more general and also involves key vectors  kiki: the attention
> weights  pp  are actually computed between the keys and the lookup,
> and the context is then constructed with the  vivi.
## 注意力机制（attention mechanism）
基于组成整体的各个元素在整体中发挥的作用不相同这样一个事实，注意力机制的基本思想是通对使用不同的权重组合各个序列元素来描述整体，这就好像我们在快速观察人物的照片时会把注意力更多的放在人物的面部而几乎不会留意背景中的某一棵小草。本质上来讲，注意力机制是对组成整体的元素加权求和的过程。权值的计算方法由任务目标来确定，这就好像。。。对。。。的关注程度不一致是一个道理。在机器翻译（一种常见的seq2seq任务）中一种常见的权值衡量方法是计算序列元素（单词）之间的相似度。
注意力机制主要用于seq2seq任务，它的基本思想就是将人类快速阅读的方式应用在序列分析上。不同于RNN中先通过依次分析输入元素来逐步生成context vector的方式，注意力机制对这些输入元素进行加权平均的方式来一步生成context vector。这样做的好处不仅大大加速了context vector的生成，而且避免了RNN的长序列训练困难的问题。
-   **首先**，从数学公式上和代码实现上Attention可以理解为**加权求和**。
-   **其次**，从形式上Attention可以理解为**键值查询**。
-   **最后**，从物理意义上Attention可以理解为**相似性度量**。
图attention mechanism
![enter image description here](https://oscimg.oschina.net/oscnet/5bdc25e12070e665409112ee13ac9e76603.jpg)

![公式](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWicMiaqpI5cdFEvj2sOZVykZic5SwVXksjias1lA5ukFcJ4ficRgmwIyBLK8PcibmvT8Tq4iaIqMl0IaQfVw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
let's think about how to design a seq-seq model with only attention.


- bear minimal core model
- ![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vSqvRProLcHw7CSGx4TGpXvLAKwMiy6dEfUa1vnlhQqkx0liKnOUmE1Bm-LQogtMfT1L44jo8y_wjPB/pub?w=883&h=458)
	- embedding -> attention -> 
- add position into model
	- embedding -> positional encoding -> attention -> Dense -> softmax
- add self-attention
	- embedding -> positional encoding -> self-attention -> encoder-decoder attention -> self-attention ->
- fix attention side affect (average)
	- embedding -> positional encoding -> mutiple-headed attention -> FFN -> encoder-decoder attention -> Dense -> softmax
- 
## Transformer模型
基于attention机制
- 解决long memory problem
- 实现了部分并行运算，极大缩短了训练时间
- 提高了准确率

### 模型架构
整体架构上看，transformer仍属于Encoder-Decoder架构，通过encoder将输入序列转换成内部表示，在通过不同decoder实现不同的预测功能。
![enter image description here](http://armancohan.com/img/transformer-1.png)
Transformer的最大的创新在于它使用只attention机制来实现seq2seq task，避免使用RNN和CNN从而使得在训练速度和准确率上全面超越了已有的方法。具体来讲
![enter image description here](https://3.bp.blogspot.com/-aZ3zvPiCoXM/WaiKQO7KRnI/AAAAAAAAB_8/7a1CYjp40nUg4lKpW7covGZJQAySxlg8QCLcBGAs/s640/transform20fps.gif)

#### Attention
Attention是transformer的核心，它不仅作用在encoder到docoder的转换中，还被用在encoder和decoder内部，也被称为self-attention。
#### 自注意力（self attention）
时序问题（特别是NLP问题）中的序列元素表示的含义通常不止该单个元素的的字面意义，而是与整个序列上下文有关系，因此在encoding过程中需要考虑整个序列来决定其中每个元素的意义。self-attention机制就是基于这种由全局确定局部的思想，简单来说它使用整个序列所有元素的**加权**平均来确定每一个元素在所处序列（上下文）中的含义。
![enter image description here](https://miro.medium.com/max/410/1*NlQPdpNY4d26l8Vu92a0Wg.png)
Scaled Dot-Product Attention
其中的权值来自该元素与其他元素的相似度，这是基于这样的假设-相似度越高的元素对确定该元素在整个序列中的含义的贡献度越大，由于序列元素以向量表示（word4vec），在transformer中使用点积运算来确定相似度，其结果是一个数值。
comparison with RNN and CNN
- less complex
- can be paralleled, faster
- easy to learn distant dependency

![enter image description here](http://www.c-jump.com/bcc/common/Talk3/Math/Vectors/const_images/v06_dot.png)

> *self-attention层的好处是能够一步到位捕捉到全局的联系，解决了长距离依赖，因为它直接把序列两两比较（代价是计算量变为 O(n2)，当然由于是纯矩阵运算，这个计算量相当也不是很严重），而且最重要的是可以进行并行计算。 相比之下，RNN
> 需要一步步递推才能捕捉到，并且对于长距离依赖很难捕捉。而 CNN 则需要通过层叠来扩大感受野，这是 Attention 层的明显优势。*
self-attention其实和cnn，rnn一样，也是为了对输入进行编码，为了获得更多的信息。所以应把self-attention也看成网络中的一个层加进去。

> 对于使用自注意力机制的原因，论文中提到主要从三个方面考虑（每一层的复杂度，是否可以并行，长距离依赖学习），并给出了和RNN，CNN计算复杂度的比较。可以看到，如果输入序列n小于表示维度d的话，每一层的时间复杂度self-attention是比较有优势的。当n比较大时，作者也给出了一种解决方案self-attention（restricted）即每个词不是和所有词计算attention，而是只与限制的r个词去计算attention。在并行方面，多头attention和CNN一样不依赖于前一时刻的计算，可以很好的并行，优于RNN。在长距离依赖上，由于self-attention是每个词和所有词都要计算attention，所以不管他们中间有多长距离，最大的路径长度也都只是1。可以捕获长距离依赖关系。
> In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention.

平均是指——————
在transformer中的encoder和decoder中都使用了自注意力机制，他们的实现基本相同，稍有不同的是在decoder中使用mask来*屏蔽当前元素之后的元素*
#### encoder-decoder attention

![enter image description here](https://cntk.ai/jup/cntk204_s2s2.png)新问题
- 位置编码Positional encoding
![enter image description here](https://www.researchgate.net/publication/327068570/figure/fig3/AS:660457148928000@1534476663109/The-original-positional-encoding-used-in-Attention-Is-All-You-Need-VSP-17-composed.png)
![enter image description here](https://www.d2l.ai/_images/output_transformer_ee2e4a_21_0.svg)
由于transformer不使用RNN和CNN，仅仅计算不同元素之间的相似度，因此必须加入位置信息来保证transformer正确的理解输入序列。最简单的位置编码是直接使用元素的序号，但这种方式对输入序列的长度过于敏感，对相对位置关系的表达——————。 extrapolate training samples
Transformer中使用了sin/cos位置编码
	1. 计算方便
	2. 能够体现相对位置关系
	3. 可处理变长序列
### 多头注意力（ Multiple Headed Attention)
![enter image description here](https://miro.medium.com/max/600/1*Vb9UizPn0AHejEYW9CWxNQ.png)
different random initial weights matrix may lead to different representation subspace, thus give transformer ability to understand different meaning of a word
- stack of encoder/decoder layer
	- - 位置编码PosiStacking of encoder/decoder
	- self attentional， encoding
由于transformer不使用RNN 和CNN free - help to speed up training
	- Stacking of encoder/decoder
	- sel-decoding attention
- **multi-head attention** VS convolution on multiple channels
	- Convolution: Different linear transformations by relative position
	- MHA: a weighted average 
	- It is found empirically that multi-head attention works better than the usual “single-head” in the context of machine translation. And the intuition behind such an improvement is that “multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions”
### Why multiple layer of attention layers?
### Positional encoding
- why not positional index? 
### point-wise FFN
point-wise 对序列中每个元素分别进行2层全连接运算
> Like the name indicates, this is a regular feedforward network applied to _each_ time step of the Multi Head attention outputs. The network has three layers with a non-linearity like ReLU for the hidden layer. You might be wondering why do we need a feedforward network after attention; after all isn’t attention all we need 😈 ? I suspect it is needed to improve model expressiveness. As we saw earlier the multi head attention partitioned the inputs and applied attention independently. There was only a linear projection to the outputs, i.e. the partitions were combined only linearly. The _Positionwise Feedforward_ network thus brings in some non-linear ‘mixing’ if we call it that. In fact for the sequence tagging task we use convolutions instead of fully connected layers. A filter of width 3 allows interactions to happen with adjacent time steps to improve performance.
### Mask
> -   In the encoder and decoder: To zero attention outputs wherever there is just padding in the input sentences.
> -   In the decoder: To prevent the decoder ‘peaking’ ahead at the rest of the translated sentence when predicting the next word.

由于attention机制可以看到全部输入，所以需要mask来防止attention在训练时看到正确的输出 
> We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position ii can depend only on the known outputs at positions less than ii.
> I mentioned I would cover attention bias mask later when going through the code of  `MultiHeadAttention`. For tasks like translation the decoder is fed previous outputs as input to predict the next output. During training the quick way to get the previous outputs is to  _shift_  the training labels right (The first time step gets a special symbol) and feed them as decoder inputs — a technique known as  _Teacher Forcing_  in machine learning parlance. However this presents a problem for the Transformer decoder as it can ‘cheat’ by using inputs from future time steps. The places where the short circuiting can happen is the self attention step and both the feedforward steps. (Can you figure out why it cannot happen in the normal attention step?)

> In the self attention step we feed values from all time steps to the  `MultiHeadAttention`  component. Recall that we do a weighted linear combination of the  _Values_  input:

![](https://miro.medium.com/max/504/1*aJiWfOaTCktprHEgNdeJow.png)

Consider the first row of  _OUTPUT_  in the above diagram. It corresponds to the attention output at time  _t=1_. But it is computed from values right up till  _t=10_  which are future time steps. To prevent reading these future values we zero out all weights in the  _WEIGHTS_  tensor above the main diagonal. This will ensure that future values cannot creep in:

![](https://miro.medium.com/max/204/1*6aTQQSmXUfCQxj3drNEweg.png)
## Transformer的改进
Despite not having any explicit recurrency, implicitly the model is built as an autoregressive one. It implies that in order to generate an output (both while training or during inference), the model needs to compute previous outputs, which is extremely costly, for the whole net has to be run for every output. That’s the main idea to overcome in a recent paper by researchers at [_Salesforce Research_](https://einstein.ai/research/non-autoregressive-neural-machine-translation) and the University of Hong Kong, who tried to make the whole process parallelizable[23](https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html#fn:23). Their proposal is to compute _fertilities_ for every input word in the sequence, and use it instead of previous outputs in order to compute the current output. This is summarized in the figure below.
## Transformer实现
### layer normalization
### residual connection
- Help gradient propagated back through stacked decoders and encoders
- Residuals carry positional information to higher layers, among other information.
### warn-up learning rate
### regularization
- dropout
- layer normalization
## 总结

## Resources
[Attention is all you need review]([https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html](https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html))
[The transformer - Attention is all you need]([https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XTEl6ugzZPY](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XTEl6ugzZPY))
[Building the Mighty Transformer for Sequence Tagging in PyTorch](https://medium.com/@kolloldas/building-the-mighty-transformer-for-sequence-tagging-in-pytorch-part-i-a1815655cd8](https://medium.com/@kolloldas/building-the-mighty-transformer-for-sequence-tagging-in-pytorch-part-i-a1815655cd8))
[Walkthrough: The Transformer Architecture](https://www.lesswrong.com/posts/qscAeYE67GoSffDDA/walkthrough-the-transformer-architecture-part-1-2)
[The Transformer: Attention Is All You Need](https://glassboxmedicine.com/2019/08/15/the-transformer-attention-is-all-you-need/)
[How to code The Transformer in PyTorch](https://blog.floydhub.com/the-transformer-in-pytorch/)
[https://www.d2l.ai/chapter_attention-mechanism/transformer.html](https://www.d2l.ai/chapter_attention-mechanism/transformer.html)
[What is a Transformer?](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)
[Paper Dissected: “Attention is All You Need” Explained](https://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/)
[https://docs.dgl.ai/en/latest/tutorials/models/4_old_wines/7_transformer.html](https://docs.dgl.ai/en/latest/tutorials/models/4_old_wines/7_transformer.html)
[https://www.tensorflow.org/beta/tutorials/text/transformer#point_wise_feed_forward_network](https://www.tensorflow.org/beta/tutorials/text/transformer#point_wise_feed_forward_network)
[Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#a-family-of-attention-mechanisms)
[The Transformer – Attention is all you need.](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/)
[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
[Create The Transformer With Tensorflow 2.0](https://machinetalk.org/2019/04/29/create-the-transformer-with-tensorflow-2-0/)
[深度学习中的注意力机制](https://blog.csdn.net/songbinxu/article/details/80739447)
<!--stackedit_data:
eyJoaXN0b3J5IjpbODI3MjkxMDE1LDE5NjA4MDI5MDUsODExNT
EyMjEyLC05MTA4OTM1NzgsMzc2NTE1NjYyLDEwODc4MzA2NjYs
NjQzODMwNjUsNTE2NTAyOTY1LC01NTQzOTUzNTYsLTk5MzcyMj
Q1NCwzMTc1MjEwMDYsNTA4NzU1NTU3LDE1NzgzNDA4NTksLTE5
OTI3MTYxODMsMTE5NzUyNzMwMCw5MTU1NTYyNjEsLTI0Nzc3MT
Q0MywxNzcxNDYxOTMwLC0xMDAxNTMwMjEzLC00MjEzNzU1MjNd
fQ==
-->