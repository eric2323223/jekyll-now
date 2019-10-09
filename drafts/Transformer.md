# Transformer-设计和构建高效的时序模型
在自然语言处理(NLP)领域，RNN一直是被最广泛使用的深度机器学习模型，近年来CNN也逐渐被用于进行。。。然而这两类模型都有一些难以克服的问题，Transformer就是为了解决这些问题的新型模型，并取得了非常好的效果，大有取代RNN在NLP领域的统治地位的趋势，本文我们就来一步步的分析和理解这个优秀的seq2seq模型。

## 序列到序列问题（seq2seq）
seq2seq问题是使用机器学习（特别是深度学习）解决的一类常见问题，例如机器翻译，语态分析，摘要生成等自然语言处理问题（NLP），还包括_______。 这类问题的最大特点是输入（或输出）以序列的形式出现，序列的长度可变，任务通常要求分析整个序列才能产生输出————————。使用机器学习（深度学习）处理seq2seq任务，通常使用编码器-解码器（encoder-decoder）架构，编码器负责将输入序列转换为整个序列的内部表示（context vector），解码器则对这个内部表示进行解释。
![enter image description here](https://img-blog.csdn.net/20180627114128329?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hwdWxmYw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)传统上有两类模型：
- RNN
处理seq2seq问题的传统方法是使用RNN模型，RNN能够保存状态，它将输入分为多步，依靠每步输入和上一步的状态更新当前的状态（和输出），通过重复这种步骤在读入所有序列元素后得到整个序列的内部表示（context vector）。![enter image description here](https://miro.medium.com/max/2658/1*Ismhi-muID5ooWf3ZIQFFg.png)
从模型结构上来说特别适合序列到序列问题。问题有三点
1. 长序列的训练很困难，梯度下降算法在长序列的训练中容易发生梯度爆炸或梯度消失，虽然LSTM可以改善这个问题，但是在较长序列的训练中仍然无法完全避免。
2. 只能顺序执行，训练速度很慢
3. 固定的存储不适合长序列
- CNN
CNN可以同时处理序列中的所有元素，但是由于卷积运算的视域有限，一次卷积操作只能处理有限的元素，对于较长的序列无法处理。解决办法是通过堆叠多层卷积操作来逐渐增加视域，但这样会不可避免的导致信息丢失，并且仍没有完全解决长序列输入的处理问题，————————而且增加了模型的复杂度，使运算变慢，这和初衷不符。

总结一下，上述两种模型对于长序列的处理都有缺陷。RNN需要一步一步的处理输入序列，CNN做出了一些改进但并不彻底。从根本上的解决这个问题需要能一次性的处理全部输入（无论序列有多长），并且能根据这些输入信息分析序列元素之间的关联关系。人们从自己快速浏览的方式获得了启发，当人们需要快速浏览的时候不会按输入的顺序逐步阅读，而会直接跳到需要关注的的部分，这种根据需要在不同位置跳跃的阅读方式和注意力相关，因此这种新的序列处理方式被命名为注意力机制
![enter image description here](https://www.visionears.nl/images/babyproduct.jpg)
Attention机制来自于人类视觉注意力机制。人们视觉在感知东西的时候一般不会是一个场景从到头看到尾每次全部都看，而往往是根据需求观察注意特定的一部分。而且当人们发现一个场景经常在某部分出现自己想观察的东西时，人们会进行学习在将来再出现类似场景时把注意力放到该部分上。
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
基于组成整体的各个元素在整体中发挥的作用不相同这样一个事实，注意力机制的基本思想是通对使用不同的权重组合各个序列元素来描述整体，~~这就好像我们在快速观察人物的照片时会把注意力更多的放在人物的面部而几乎不会留意背景中的某一棵小草~~。从数学运算来讲，注意力机制是对组成整体的元素加权求和的过程。权值的计算方法由任务目标来确定，这就好像。。。对。。。的关注程度不一致是一个道理。在机器翻译（一种常见的seq2seq任务）中一种常见的权值衡量方法是计算序列元素（单词）之间的相似度。
注意力机制最早使用在基于[RNN的机器翻译模型](https://arxiv.org/pdf/1409.0473.pdf)中，不同于以往使用固定的context vector， attention能够让解码器每次解码的时候关注更相关的输入元素（生成动态的context vector）从而提高翻译的准确度。

$$c_i=\sum_{j=1}\alpha_{ij}h_j$$
$$\alpha_{ij}=\frac{exp(e_{ij})}{\sum_{k=1}exp(e_{ik})}$$
$$e_{ij}=alignment(h_i,x_j)$$

![enter image description here](https://oscimg.oschina.net/oscnet/5bdc25e12070e665409112ee13ac9e76603.jpg)

注意力机制主要用于seq2seq任务，它的基本思想就是对序列中的每个元素以一定的规则加入上下文信息。不同于RNN中先通过依次分析输入元素来逐步生成上下文context vector的方式，注意力机制对这些输入元素进行加权平均的方式来一步加入所有元素信息来生成上下文context vector。这样做的好处是能够一步到位捕捉到全局的联系(序列元素直接进行两两比较),不仅大大加速（可以并行计算）了context vector的生成，而且避免了RNN的长序列训练困难的问题。
![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vQZ5I4YZtpZOU8xnxqqJ2WVd7o9eeo0sHQa119cWm4qR85KanMs7-Z1DV1EfKxJLQrZaVglHLUJGPF2/pub?w=856&h=225)
-   **首先**，从数学公式上和代码实现上Attention可以理解为**加权求和**。
- 从实现上来讲，attention操作可以理解为一个jia'q
-  **本质**，***对元素在序列的上下文环境中重定义*** 
-   **其次**，从形式上Attention可以理解为**键值查询**。
![enter image description here](https://ldzhangyx.github.io/2018/10/14/self-attention/1.jpg)
-   **最后**，从物理意义上Attention可以理解为**相似性度量**。


Attention is cheap, 特别适合机器翻译的场景（dim>length）
||FLOPs|
|--|--|
| attention | $O(length^2 \cdot dim)$ |
| RNN | $O(length \cdot dim^2)$ |
| CNN | $O(length \cdot dim^2 \cdot kernelwidth)$ |
由于通常dim要大于length，所以self-attention的运算量会少于RNN和CNN，

 
## Transformer模型
基于attention机制
- 解决long memory problem
- 实现了部分并行运算，极大缩短了训练时间
- 提高了准确率
![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vSBNAHsyf_HP3_CkV1cygicnt0LhGxWcvw2PofecPP9TYJj41bghsAXTM6l6OSonSMvAjjgFInVDxC4/pub?w=961&h=590)

### 模型架构
整体架构上看，transformer仍属于Encoder-Decoder架构，通过encoder将输入序列转换成内部表示，在通过不同decoder实现不同的预测功能。

Transformer的最大的创新在于它使用只attention机制来实现seq2seq task，避免使用RNN和CNN从而使得在训练速度和准确率上全面超越了已有的方法。具体来讲
![enter image description here](https://3.bp.blogspot.com/-aZ3zvPiCoXM/WaiKQO7KRnI/AAAAAAAAB_8/7a1CYjp40nUg4lKpW7covGZJQAySxlg8QCLcBGAs/s640/transform20fps.gif)

### 为什么Attention is all you need?
Attention是transformer的核心，它不仅作用在encoder到docoder的转换中，还被用在encoder和decoder内部，也被称为self-attention。
- encoder-decoder attention
- encoder attention
- decoder attention
#### 自注意力（self attention）
时序问题（特别是NLP问题）中的序列元素表示的含义通常不止该单个元素的的字面意义，而是与整个序列上下文有关系，因此在encoding过程中需要考虑整个序列来决定其中每个元素的意义。self-attention机制就是基于这种由全局确定局部的思想，简单来说它使用整个序列所有元素的**加权**平均来确定每一个元素在所处序列（上下文）中的含义。
在encoder-decoder模型中encoder负责将输入转化为输入序列的内部表示（context vector），传统方法使用RNN通过一步步的叠加分析过的输入来得到整个序列的内部表示（固定长度），Transformer模型中使用自注意力（self attention）机制来实现encoding，之所以称作自注意力是因为这是在输入序列内部进行的attention操作，由于attention操作就是对元素进行重新定义使其包含序列上下文信息，在输入序列元素进行attention的操作结果就是使该元素包含输入序列信息，因此经过self attention运算的整个输入序列的结果就是和一个输入序列大小一致的context vector。显然，self attention不需要想RNN那样一步步的出入输入，而是可以同时对每个元素进行attention运算，如图所示
![enter image description here](!%5Benter%20image%20description%20here%5D%28https://docs.google.com/drawings/d/e/2PACX-1vQZ5I4YZtpZOU8xnxqqJ2WVd7o9eeo0sHQa119cWm4qR85KanMs7-Z1DV1EfKxJLQrZaVglHLUJGPF2/pub?w=856&h=225%29)
> 为什么需要在encoder中做self-attention
> ”`The animal didn't cross the street because it was too tired`”
> What does “it” in this sentence refer to? Is it referring to the street or to the animal? It’s a simple question to a human, but not as simple to an algorithm.

> 对于使用自注意力机制的原因，论文中提到主要从三个方面考虑（每一层的复杂度，是否可以并行，长距离依赖学习），并给出了和RNN，CNN计算复杂度的比较。可以看到，如果输入序列n小于表示维度d的话，每一层的时间复杂度self-attention是比较有优势的。当n比较大时，作者也给出了一种解决方案self-attention（restricted）即每个词不是和所有词计算attention，而是只与限制的r个词去计算attention。在并行方面，多头attention和CNN一样不依赖于前一时刻的计算，可以很好的并行，优于RNN。在长距离依赖上，由于self-attention是每个词和所有词都要计算attention，所以不管他们中间有多长距离，最大的路径长度也都只是1。可以捕获长距离依赖关系。
> In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention.

> Authors motivates the use of self-attention layers instead of recurrent or convolutional layers with three desiderata:

1.  Minimize total computational complexity per layer
    
    -   **Pros:**  self-attention layers connects all positions with  O(1)O(1)  number of sequentially executed operations (eg. vs  O(n)O(n)  in RNN)
2.  Maximize amount of parallelizable computations, measured by minimum number of sequential operations required
    
    -   **Pros:**  for sequence length  nn  < representation dimensionality  dd  (true for SOTA sequence representation models like  _word-piece, byte-pair_). For very long sequences  n>dn>d  self-attention can consider only neighborhood of some size  rr  in the input sequence centered around the respective output position, thus increasing the max path length to  O(n/r)O(n/r)
3.  Minimize maximum path length between any two input and output positions in network composed of the different layer types . The shorter the path between any combination of positions in the input and output sequences, the easier to learn long-range dependencies. (See why  [Hochreiter et al, 2001](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.24.7321) ) 

**Scaled Dot-Product Attention**
其中的权值来自该元素与其他元素的相似度，这是基于这样的假设-相似度越高的元素对确定该元素在整个序列中的含义的贡献度越大，由于序列元素以向量表示（word4vec），在transformer中使用点积运算来确定相似度，其结果是一个数值。形式化的定义为
$W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$
$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$
![enter image description here](https://miro.medium.com/max/410/1*NlQPdpNY4d26l8Vu92a0Wg.png)


![enter image description here](http://www.c-jump.com/bcc/common/Talk3/Math/Vectors/const_images/v06_dot.png)
![enter image description here](https://miro.medium.com/max/1452/1*oosK1XGaYr0AoSxfs9fx5A.png)

在transformer中的encoder和decoder中都使用了自注意力机制，他们的实现基本相同，稍有不同的是在decoder中使用mask来*屏蔽当前元素之后的元素*
#### encoder-decoder attention
In terms of encoder-decoder, the **query** is usually the hidden state of the _decoder_. Whereas **key**, is the hidden state of the _encoder_, and the corresponding **value** is normalized weight, representing how much attention a _key_ gets. Output is calculated as a wighted sum – here the dot product of _query_ and _key_ is used to get a _value_.

![enter image description here](https://cntk.ai/jup/cntk204_s2s2.png)

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

### 位置编码（positional encoding）
与RNN和CNN不同，在Attention中没有词序的概念（如第一个词，第二个词等）， 输入序列的所有单词都以没有特殊顺序或位置的方式输入网络，因此模型不知道单词的顺序。 因此，需要将与位置相关的信号添加到每个词中，以帮助模型理解词的顺序。
位置编码是单词值及其在句子中位置的重新表示（假定开头和结尾或中间的开头和开头不相同）。考虑到句子的长度可以是任意长度，只讨论词的绝对位置是不全面的（同一个词，在由3个词组成的句子中的第三个位置和30个词组成的句子中的第三个位置所表达的意思很可能是不一样的）。位置编码器的作用是获得sin（x）和cos（x）函数的循环特性的帮助，以返回单词在句子中的位置信息。
> 通常，将位置编码添加到输入嵌入是一个非常有趣的话题。一种方法是嵌入输入元素的绝对位置（如在ConvS2S中一样）。但是，作者使用“不同频率的正弦和余弦函数”。 “正弦波”版本非常复杂，同时具有与绝对位置版本相似的性能。然而，问题的关键在于，它可以使模型在测试时对更长的句子产生更好的翻译（至少比训练数据中的句子更长）。通过这种正弦方法，模型可以外推到更长的序列长度3。

由于attention机制不考虑位置关系，因此必须要在在attention操作前对序列中的每个元素加入位置信息。一个最直接的方法就是对输入加入序号，但是这种方法的问题在于无法处理长度超过训练数据的输入序列。在Transformer模型中使用的是sin/cos函数进行位置编码，主要目的是利用sin/cos函数的周期性来进行任意长度序列的位置编码。

sin/cos embedding has 2 advantage
	- always between -1 and 1, 非常有利于梯度下降算法的计算
	- most benefit of relative positional encoding is it can work with any size of the input, which is longer than the longest of the training data
$$PE_{{pos,2i}}=sin(pos/10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)}=cos(pos/10000^{2i/d_{model}})$$

为何采用叠加的方式？

> 直觉是，在高维中随机选择的向量几乎总是近似正交的。没有理由认为单词向量和位置编码向量之间有任何关联。如果单词嵌入形成一个较小维的子空间，而位置编码形成另一个较小维的子空间，则两个子空间本身可能近似正交，因此大概可以对这些子空间进行变换，尽管进行了矢量相加，但两个子空间仍可以通过一些单个学习的变换而彼此独立地进行操作。因此，串联并不会增加太多，但会大大增加学习参数方面的成本。


![enter image description here](https://www.researchgate.net/publication/327068570/figure/fig3/AS:660457148928000@1534476663109/The-original-positional-encoding-used-in-Attention-Is-All-You-Need-VSP-17-composed.png)


### 多头注意力（ Multiple Headed Attention)
![enter image description here](https://miro.medium.com/max/600/1*Vb9UizPn0AHejEYW9CWxNQ.png)
Transformer仅仅使用attention进行输入encoding，由于attention本质上只是对输入进行加权平均运算，这导致特征提取能力不足，为了解决这个问题作者提出了多头注意力（）的方法。多头注意力的基本思想通过多次初始化过程增加模型提取不同特征的机会，假设下图中通过三次初始化分别得到了三种特征：红色表示动作，绿色表做动作施加者，蓝色表示动作承受着，可以看到在对“踢“进行了三次self attention运算，分别对应三种特征。在对于动作信息的self attention中，"我“和”球“的权值（灰色细线表示）比“踢”的权值（红色粗线）要小很多；同样，对动作施加者的self attention中，“我”（绿色粗线）则是主要贡献者。在将三次self attention的结果相加后，得到的新的“踢”的编码中就包含了三种特征的信息。现实中不可能每次随机初始化都能带来有效的特征，理论上随机初始化测次数越多就越有可能发现有效的特征，不过随之增长的是训练参数的增加，这意味着训练难度的提高，因此需要平衡，再Transformer模型中这个值是8。
具体方法是对同一个元素进行多次attention运算， 每次attention都使用不同的初始化参数W，最后在将多次attention的结果相加。
![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vT4_Vn34rr1zN4OhXIo7oCGkzXDF__Y3CIVnZ_12fjqLHtKoRSJaVIyoR7ndQHtRlfNUmgecF5mucNg/pub?w=538&h=363)
> In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention.

different random initial weights matrix may lead to different representation subspace, thus give transformer ability to understand different meaning of a word
- **multi-head attention** VS convolution on multiple channels
	- Convolution: Different linear transformations by relative position
	- MHA: a weighted average 
	- It is found empirically that multi-head attention works better than the usual “single-head” in the context of machine translation. And the intuition behind such an improvement is that “multi-head attention allows the model to jointly attend to information from **different representation subspaces at different positions**”

> Transformer reduces the number of operations required to relate (especially distant) positions in input and output sequence to a O(1)O(1). However, this comes at cost of reduced effective resolution because of averaging attention-weighted positions.
> To reduce this cost authors propose the multi-head attention:
> Transformer use multi-head (dmodel/hdmodel/h  parallel attention functions) attention instead of single (dmodeldmodel-dimensional) attention function (i.e.  q,k,vq,k,v  all  dmodeldmodel-dimensional). It is at similar computational cost as in the case of single-head attention due to reduced dimensions of each head.
> Transformer imitates the classical attention mechanism (known e.g. from  [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) or Conv2S2) where in encoder-decoder attention layers  _queries_  are form previous decoder layer, and the (memory)  _keys_  and  _values_  are from output of the encoder. Therefore, each position in decoder can attend over all positions in the input sequence.

### point-wise FFN
point-wise 对序列中每个元素分别进行2层全连接运算，目的主要是为了提供对multi-attention提取出的feature进行 **复杂（非线性）** 合成的能力
> Like the name indicates, this is a regular feedforward network applied to _each_ time step of the Multi Head attention outputs. The network has three layers with a non-linearity like ReLU for the hidden layer. You might be wondering why do we need a feedforward network after attention; after all isn’t attention all we need 😈 ? I suspect it is needed to improve model expressiveness. As we saw earlier the multi head attention partitioned the inputs and applied attention independently. There was only a linear projection to the outputs, i.e. the partitions were combined only linearly. The _Positionwise Feedforward_ network thus brings in some non-linear ‘mixing’ if we call it that. In fact for the sequence tagging task we use convolutions instead of fully connected layers. A filter of width 3 allows interactions to happen with adjacent time steps to improve performance.

### Why multiple layer of attention layers?





## Transformer优化技巧
由于Transformer的encoder和decoder各自都由若干个encoder/decoder层组成（每个encocer/decoder又由一multihead attention和两层Feed foward network构成），属于比较复杂的模型，因此要通过使用一些优化技巧才能进行训练。
### 残差链接(residual connection)
残差链接可以算得上深度学习中的神器，特别适合用于深度神经网络模型的训练。它的基本思想是
在Transformer中
- Help gradient propagated back through stacked decoders and encoders
- Residuals carry positional information to higher layers, among other information.

### regularization
- dropout
- layer normalization
- label smoothing
### 超参数（hyperparameter tunning）
- warn-up learning rate
> If your data set is highly differentiated, you can suffer from a sort of "early over-fitting". If your shuffled data happens to include a cluster of related, strongly-featured observations, your model's initial training can skew badly toward those features -- or worse, toward incidental features that aren't truly related to the topic at all. Warm-up is a way to reduce the primacy effect of the early training examples. Without it, you may need to run a few extra epochs to get the convergence desired, as the model un-trains those early superstitions.
> Many models afford this as a command-line option. The learning rate is increased linearly over the warm-up period. If the target learning rate is  `p`  and the warm-up period is  `n`, then the first batch iteration uses  `1*p/n`  for its learning rate; the second uses  `2*p/n`, and so on: iteration  `i`  uses  `i*p/n`, until we hit the nominal rate at iteration  `n`.
> This means that the first iteration gets only 1/n of the primacy effect. This does a reasonable job of balancing that influence.
> Note that the ramp-up is commonly on the order of one epoch -- but is occasionally longer for particularly skewed data, or shorter for more homogeneous distributions. You may want to adjust, depending on how functionally extreme your batches can become when the shuffling algorithm is applied to the training set.

在介绍了Transformer的主要组成部分之后，我们再来完整看一下Transformer模型
![enter image description here](https://camo.githubusercontent.com/4b80977ac0757d1d18eb7be4d0238e92673bfaba/68747470733a2f2f6c696c69616e77656e672e6769746875622e696f2f6c696c2d6c6f672f6173736574732f696d616765732f7472616e73666f726d65722e706e67)
## Transformer的改进
Despite not having any explicit recurrency, implicitly the model is built as an autoregressive one. It implies that in order to generate an output (both while training or during inference), the model needs to compute previous outputs, which is extremely costly, for the whole net has to be run for every output. That’s the main idea to overcome in a recent paper by researchers at [_Salesforce Research_](https://einstein.ai/research/non-autoregressive-neural-machine-translation) and the University of Hong Kong, who tried to make the whole process parallelizable[23](https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html#fn:23). Their proposal is to compute _fertilities_ for every input word in the sequence, and use it instead of previous outputs in order to compute the current output. This is summarized in the figure below.
尽管没有任何显式递归，但是隐式地将模型构建为自回归模型。 这意味着为了生成输出（在训练时或在推理期间），该模型需要计算先前的输出，这非常昂贵，因为必须为每个输出运行整个网络。 这是Salesforce Research和香港大学的研究人员在最近的一篇论文中要克服的主要思想，他们试图使整个过程可并行化23。 他们的建议是为序列中的每个输入单词计算肥力，并使用它代替先前的输出以计算当前输出。 下图对此进行了总结。
![enter image description here](https://ricardokleinklein.github.io/images/transformer/fertilities.png)
## 总结
Transformer不是万能的，它在NLP领域取得突破性成绩是由于它针对机器翻译领域做了针对性的设计，比如positional enbemdding， self attention， multihead attention，并结合了多种相关的优化技巧，如residual connection，layer normalization等。
因此，对于任何任务，都需要针对任务目标进行相对应设计，并且要进行优化才能充分发挥模型的优势。
一个好的模型不会从天而降，而是需要不断地分析觉接问题才能逐渐完善，通过对Transformer的学习，也可以掌握对已有模型进行改进的基本思路，1. 找到痛点并针对主要问题进行设计；2. 建立核心模型后要对随之产生的新问题提出解决方案；3.通过实验进行验证，还有利用已有的优化方法进行优化。

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
[nlp中的Attention注意力机制+Transformer详解](https://zhuanlan.zhihu.com/p/53682800)
[Attention and its Different Forms](https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc)
[Attn: Illustrated Attention](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
[https://mchromiak.github.io/articles/2017/Sep/01/Primer-NN/#attention-basis](https://mchromiak.github.io/articles/2017/Sep/01/Primer-NN/#attention-basis)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIzNTIwNjI4MCwtMjAwNDQxODgzMCwtMj
AxNjYyMTAyNywtOTEyNTg1NzY0LDIwNzU3NjIwOSwxNTEwODg1
NDMxLDIwNTIzOTE3OTAsLTEzMjQzMDYyNzAsLTEzNTQyNjQ4OT
EsMTU2MzYxMTE3OCwxMTYzMDAwOTQwLC02MzE4MzAzOTgsMTIy
MjA1MDA4LC0zNTA0Mzc1NDQsLTk4OTQ1MzkwOCwtMTQ2NzIxMT
Y3NiwxNjAzNTgwNjI1LC0yNzE1NTM1NDUsMTkyMTE0MDA5NCwt
NjQ4MzM1NzZdfQ==
-->