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
从实现上来讲，attention操作可以理解为加权求和的运算，加数是序列中的所有元素，权值计算方法根据任务目标而不同（在机器翻译的场景中使用相似度来作为权值）。用$\alpha$表示权值（通常表现为概率分布，即$\sum \alpha=1$），$h$表示序列元素，可以将attention形式化的表示为
$$y_2=w_{21}x_1+w_{22}x_2+w_{23}x_3+w_{24}x_4$$
$$y_i=\sum_{j=1}w_{ij}x_j$$
![enter image description here](http://www.peterbloem.nl/files/transformers/self-attention.svg)
从这个定义可以看出attention的结果$c$就是序列中所有元素按一定的比例关系相加得到的，由于$c$具备了序列的上下文信息，因此我们也可以把attention理解为**元素在某一个序列上下文环境中的重新定义**。这是attention最核心的特点，也是attention能够取代RNN的基础。
权值$w_{ij}$表示$x_j$在对于$y_i$的计算中发挥的权重，由于所有$x$都参与$y_i$的计算，所以使用softmax来保证所有权值的和等于1。
$$w_{ij}=\frac{exp(e_{ij})}{\sum_{k=1}exp(e_{ik})}$$
这里的$e_{ij}$表示$x_j$和$y_i$的相关性，对于机器翻译任务来说，通常用矢量相似性来表述元素的相关性，适量相似性的计算方法有很多，其中最常用的就是点积运算（dot product）

$$e_{ij}=x_j\cdot y_i=|x_j||y_i|cos\theta$$ 
$\theta$表示两个向量$a,b$之间的夹角，如果$a,b$越相似则夹角$\theta$越小，$cos\theta$则越接近1

![enter image description here](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSO0ZVpogoaP-ipyQF0Xhir4wSrgGJBdeU_5wDrea6UD9sF7icIYg)
-   **最后**，从物理意义上Attention可以理解为**相似性度量**。
$$e_{ij}=Sim(h_i,x_j)$$

> **try to understand why K and V are different in transformer first!!!**
> Attention has a more generalized the form: XXXXXX
> 
> Goal is to learn $W_k, W_q, W_v$ so that 
-   **其次**，从形式上Attention可以理解为**键值查询**
对于进行相似性计算和——不同的情况，Attention可以更一般的表示为
$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(Sim(Q,K))V$$
上式表示对于查询$q$和键值对$K,V$Given a query  **q**  and a set of key-value pairs  **(K, V)**, attention can be generalised to compute a weighted sum of the values dependent on the query and the corresponding keys.  
The query determines which values to focus on; we can say that the query ‘attends’ to the values.

![enter image description here](https://ldzhangyx.github.io/2018/10/14/self-attention/1.jpg)


## Transformer模型

Transformer来自Google Brain团队2017年的文章Attention is all you need。正如论文的题目所说的，整个网络结构完全是由Attention机制组成。由于没有使用RNN和CNN，避免了无法并行计算和长距离依赖等传统方法无法克服的问题，用更少的计算资源，取得了比过去的结构更好的结果，在机器翻译中取得了BLEU值得新高。
整体架构上看，transformer仍属于Encoder-Decoder架构，通过encoder将输入序列转换成内部表示，再通过不同decoder实现不同的预测功能。从图中可以看到，编码器主要由两种组件构成：
![enter image description here](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/06/Screenshot-from-2019-06-17-20-01-32.png)


![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vSBNAHsyf_HP3_CkV1cygicnt0LhGxWcvw2PofecPP9TYJj41bghsAXTM6l6OSonSMvAjjgFInVDxC4/pub?w=961&h=590)


### 为什么Attention is all you need?
Transformer论文的标题说只需要attention意味着attention可以完成以前需要RNN才能做的工作。由于RNN有存储的能力，因此可以在编码阶段通过不断的处理和积累一个个的输入元素从而最终获得这个输入序列的上下文信息（context vector），同样在解码阶段根据context vector产生输出。在transformer模型中设计了自注意力机制来生成conext vector
> Attention是transformer的核心，它不仅作用在encoder到decoder的转换中，还被用在编码器（encoder）和解码器（decoder）内部，这种在编码解码器内部使用的attention被称为自注意力self-attention。自注意力用于替代RNN来做encoding

#### 自注意力（self attention）
> 时序问题（特别是NLP问题）中的序列元素表示的含义通常不止该单个元素的的字面意义，而是与整个序列上下文有关系，因此在encoding过程中需要考虑整个序列来决定其中每个元素的意义。self-attention机制就是基于这种由全局确定局部的思想，简单来说它使用整个序列所有元素的**加权**平均来确定每一个元素在所处序列（上下文）中的含义。

在encoder-decoder模型中encoder负责将输入转化为输入序列的内部表示（context vector），传统方法使用RNN通过一步步的叠加分析过的输入来得到整个序列的内部表示（固定长度），Transformer模型中使用自注意力（self attention）机制来实现encoding，之所以称作自注意力是因为这是在输入序列内部进行的attention操作，由于attention操作就是对元素进行重新定义使其包含序列上下文信息，在输入序列元素进行attention的操作结果就是使该元素包含输入序列信息，因此经过self attention运算的整个输入序列的结果就是和一个输入序列大小一致的context vector。显然，self attention不需要想RNN那样一步步的出入输入，而是可以同时对每个元素进行attention运算，从下图可以发现，RNN需要在依次处理元素x1, x2和x3之后才能得到整个序列的上下文信息，而attention则可以同时处理x1，x2，x3而得到序列的上下文信息。
![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vQZ5I4YZtpZOU8xnxqqJ2WVd7o9eeo0sHQa119cWm4qR85KanMs7-Z1DV1EfKxJLQrZaVglHLUJGPF2/pub?w=856&h=225)


总结来说，Attention比较RNN有一下三点优势
- 对于NLP的任务场景，attention的计算复杂度更低（dim>length）

||FLOPs|
|--|--|
| attention | $O(length^2 \cdot dim)$ |
| RNN | $O(length \cdot dim^2)$ |
| CNN | $O(length \cdot dim^2 \cdot kernelwidth)$ |
由于通常dim要大于length，所以self-attention的运算量会少于RNN和CNN，

- 在并行方面，多头attention和CNN一样不依赖于前一时刻的计算，可以很好的并行，优于RNN。
- 在长距离依赖上，由于self-attention是每个词和所有词都要计算attention，所以不管他们中间有多长距离，最大的路径长度也都只是1。可以捕获长距离依赖关系。RNN则存在梯度弥散或者梯度爆炸的问题。
#### Attention mask
Attention这种新的结构使得他的训练方式也和RNN不同，这是由于Attention可以直接看到所有的元素，因此需要mask来防止——————， 具体来看
- 编码器self attention，不需要mask
- 编码器-解码器attention，需要对padding进行mask
- 解码器self attention，需要对当前位置之后的所有元素masking

#### Scaled Dot-Product Attention (SDPA)
Transformer对标准的attention做了一个小小调整：加入特征缩放（feature scaling）。这样做主要是为了防止softmax运算将值较大的key过度放大，导致其他key的信息很难加入到attention结果中。
$$\mathrm{SDPA}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$
特征缩放体现在对$Q$和$K$计算点积$QK^T$以后，增加了一步除以$\sqrt{d_k}$运算。
下图是上式的图像化表示，其中Scale就是特征缩放的操作。

>其中的权值来自该元素与其他元素的相似度，这是基于这样的假设-相似度越高的元素对确定该元素在整个序列中的含义的贡献度越大，由于序列元素以向量表示（word4vec），在transformer中使用点积运算来确定相似度，其结果是一个数值。形式化的定义为
$W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

![enter image description here](https://miro.medium.com/max/676/1*nCznYOY-QtWIm8Y4jyk2Kw.png)


####  encoder-decoder attention
In terms of encoder-decoder, the **query** is usually the hidden state of the _decoder_. Whereas **key**, is the hidden state of the _encoder_, and the corresponding **value** is normalized weight, representing how much attention a _key_ gets. Output is calculated as a wighted sum – here the dot product of _query_ and _key_ is used to get a _value_.

### 位置编码（positional encoding）
与RNN和CNN不同，在Attention中没有词序的概念（如第一个词，第二个词等）， 输入序列的所有单词都以没有特殊顺序或位置的方式输入网络，因此模型不知道单词的顺序。 因此，需要将与位置相关的信号添加到每个词中，以帮助模型理解词的顺序。
位置编码是单词值及其在句子中位置的重新表示（假定开头和结尾或中间的开头和开头不相同）。考虑到句子的长度可以是任意长度，只讨论词的绝对位置是不全面的（同一个词，在由3个词组成的句子中的第三个位置和30个词组成的句子中的第三个位置所表达的意思很可能是不一样的）。
在Transformer模型中利用了不同频率的周期函数来进行位置编码，这种位置编码有如下优点：
- 由于sin/cos函数的周期性它能够进行任意长度序列的位置编码
- 使用多个不同频率来保证不会由于周期性导致不同位置的编码相同
- 第二是由于sin/cos函数的值总是在-1到1之间，这种编码本身也有正则化（normalization）的作用，这有利于神经网络的学习。

如果用$pos$表示位置，$i$表示元素编码的维度，$d_{model}$表示模型的维度，位置编码$PE$可以表示为
$$PE_{{pos,2i}}=sin(pos/10000^{2i/d_{model}}) $$
$$PE_{(pos, 2i+1)}=cos(pos/10000^{2i/d_{model}})$$

![enter image description here](http://vandergoten.ai/img/attention_is_all_you_need/positional_embedding.png)
计算产生的位置编码是一个与元素具有相同维度的向量，使用相加的方式将位置信息叠加进元素中，如下图所示
![enter image description here](https://wikidocs.net/images/page/31379/transformer6_final.PNG)
为何采用相加的方式？
> 直觉是，在高维中随机选择的向量几乎总是近似正交的。没有理由认为单词向量和位置编码向量之间有任何关联。如果单词嵌入形成一个较小维的子空间，而位置编码形成另一个较小维的子空间，则两个子空间本身可能近似正交，因此大概可以对这些子空间进行变换，尽管进行了矢量相加，但两个子空间仍可以通过一些单个学习的变换而彼此独立地进行操作。因此，串联并不会增加太多，但会大大增加学习参数方面的成本。

为什么要同时使用sin和cos，而不只使用其中的一个？
下图可见
![enter image description here](https://i.stack.imgur.com/5QQmq.gif)

![enter image description here](https://i.stack.imgur.com/W0b0c.gif)


### 多头注意力（ Multiple Headed Attention, MHA)

Transformer仅仅使用attention进行输入encoding，由于attention本质上只是对输入进行加权平均运算，这导致特征提取能力不足(比较convolution做线性变换，而attention只是做了加权平均)，为了解决这个问题作者提出了多头注意力（）的方法。多头注意力的基本思想通过多次初始化过程增加模型提取不同特征的机会，假设下图中通过三次初始化分别得到了三种特征：红色表示动作，绿色表做动作施加者，蓝色表示动作承受着，可以看到在对“踢“进行了三次self attention运算，分别对应三种特征。在对于动作信息的self attention中，"我“和”球“的权值（灰色细线表示）比“踢”的权值（红色粗线）要小很多；同样，对动作施加者的self attention中，“我”（绿色粗线）则是主要贡献者。在将三次self attention的结果相加后，得到的新的“踢”的编码中就包含了三种特征的信息。现实中不可能每次随机初始化都能带来有效的特征，理论上随机初始化测次数越多就越有可能发现有效的特征，不过随之增长的是训练参数的增加，这意味着训练难度的提高，因此需要平衡，再Transformer模型中这个值是8。

![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vT4_Vn34rr1zN4OhXIo7oCGkzXDF__Y3CIVnZ_12fjqLHtKoRSJaVIyoR7ndQHtRlfNUmgecF5mucNg/pub?w=538&h=363)
具体方法是对同一个元素进行多次attention运算， 每次attention都使用不同的初始化参数$W$，最后在将多次attention的结果相加。
在transformer中对每一个元素$x_i$，进行$h$次(，如word2vec,glove，后的序列元素)初始化
$$head_i =\mathrm{SDPA}(QW^Q_i, KW_i^K, VW_i^V)$$
$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(head_i, ..., head_h)W^O$$

- 对于编码器MHA，$Q, K, V$都是输入元素编码$x_i$
- 对于解码器MHA，$Q, K, V$都是已生成的输出元素编码$y_i$
- 对于编码器-解码器MHA， $Q$是输出元素编码$y_i$, $K,V$是context vector中的元素$c_i$

![enter image description here](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/img/MultiHead.png)

### 编码/解码层
transformer模型中将多头注意力HMA计算后的结果输入按位前馈网络，这里按位主要是指每个位置的元素各自输入前馈网络里进行计算，网络通常为2层，中间层维度稍大，最后一层的维度和元素编码的维度相同。这个设计的目的其实和HMA的设计类似，由于attention在特征合成能力不足，需要借助全连接网络的非线性计算来增加特征合成的能力。
需要指出的是解码层..._____________________________________
![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vTFCzc5frUSM_IkIZ9W7XE92dfKzjh9M05OqTd8FDz3mZpPBTfO0cIVQ-Uk5ZItYZGzi119CYHUaGJk/pub?w=312&h=379)![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vQPYuIriXvfFSANLnztpXorpe-MH71EMWvf0sO5EBwx1JZci48LUp6hM52ICNQ6-cga70MZe7UH6QAJ/pub?w=349&h=698)
> Like the name indicates, this is a regular feedforward network applied to _each_ time step of the Multi Head attention outputs. The network has three layers with a non-linearity like ReLU for the hidden layer. You might be wondering why do we need a feedforward network after attention; after all isn’t attention all we need 😈 ? I suspect it is needed to improve model expressiveness. As we saw earlier the multi head attention partitioned the inputs and applied attention independently. There was only a linear projection to the outputs, i.e. the partitions were combined only linearly. The _Positionwise Feedforward_ network thus brings in some non-linear ‘mixing’ if we call it that. In fact for the sequence tagging task we use convolutions instead of fully connected layers. A filter of width 3 allows interactions to happen with adjacent time steps to improve performance.

### Transformer全貌
在介绍了Transformer的主要组成部分之后，我们再来完整看一下Transformer模型。整体上来看，Transformer模型属于编码器-解码器架构，解码器需要根据序列编码sequence embedding（由编码器生成）和上一步的解码器输出来产生下一个输出，因此属于自回归(auto regressor)模型。
![enter image description here](https://camo.githubusercontent.com/4b80977ac0757d1d18eb7be4d0238e92673bfaba/68747470733a2f2f6c696c69616e77656e672e6769746875622e696f2f6c696c2d6c6f672f6173736574732f696d616765732f7472616e73666f726d65722e706e67)
编码器由若干个（N）相同的编码层堆叠形成，每个编码层主要由一个多头注意力HMA和一个按位前馈网络构成，主要作用是将序列的上下文信息融入每个元素并进行特征合成。原始的输入编码首先经过位置编码器加入位置信息，在通过多个编码层生成包含位置信息，复杂特征信息的序列编码（context vector/sequence embedding）。
解码器同样有多个（N）解码层堆叠而成。每个解码层需要两个输入，第一个输入是上一步的解码器输出（第一个解码器输出由一个固定的标识编码充当），这个输入首先要通过位置编码器加入位置信息，然后通过解码器的带遮罩的自注意力MHA（图中Masked Multi-Head Attention）加入上下文信息到已输出元素，之后加入第二个输入即序列编码，通过进行编码器-解码器MHA加入序列编码sequence embedding中的来自编码器的特征信息，最后在经过按位前馈网络合成复杂特征。经过多个解码层处理后在通过全连接运算映射到目标词典空间，最后通过softmax选择可能性最大的元素作为输出。
工作流程：
1. 输入元素进行位置编码
2. 位置编码与输入元素按位相加
3. 在编码层
	3.1 首先进行输入元素自注意力（多头注意力）计算，
	3.2 再将结果输入按位前馈网络
4. 重复多次编码层结算，结束编码阶段，得到context vector
5. 开始解码阶段，首先对输出元素进行位置编码（第一个输出为开始标记）
6. 输入元素与其位置编码按位相加
7. 在解码层
	7.1 首先进行输出元素（当前已输出）的多头自注意力计算
	7.2 进行编码（context vector）-解码（7.1结果）注意力计算
	7.3 对7.2结果输入按位前馈网络
8. 重复多次解码层计算
9. 通过全连接网络转化为目标词典宽度向量
10. 使用softmax确定输出元素（可能性最大）
11.  将当前输出元素输入6开始下一个输出元素的计算，直到输出为结束标记符

总结一下，attention是transformer的核心，它具有计算效率高（尤其对于长序列），可并行，容易训练等优势，但是同时也带了一些新问题：比如无序和特征合成能力下降。Transformer针对这些新问题分别提出了解决方案，如使用位置编码生成位置信息，使用多头注意力和按位前馈网络增强特征合成能力。

## Transformer优化技巧
由于Transformer属于比较复杂的模型，因此要通过使用一些优化技巧才能进行训练。由于Transformer中运用到的优化技术比较多，我们选择其中比较重要或者是有趣的来进行简单介绍
1. 残差链接(residual connection)
残差链接可以算得上深度学习中的神器，特别适合用于深度神经网络模型的训练。它的基本思想是
在Transformer中
   - Help gradient propagated back through stacked decoders and encoders
   - Residuals carry positional information to higher layers, among other information.

2. Layer normalization
  Normalization是在机器学习中常用的一种数据预处理方法，为了更有效的运行机器学习算法，需要将原始数据“白化”Whitening，也就是在统计学中常常提到的使数据“独立，同分布”：
	  - 独立	特征之间相关系要低
	  - 同分布	所有特征应该具有相同的均值和方差
  
   目前在深度学习中最常用的是BN，它是对不同训练数据的同一维度进行normalization，这种方法可以有效缓解深度模型训练中的梯度爆炸、弥散的问题。而在transformer采用了相对冷门的LN，主要原因是BN很难应用在训练数据长度不同的seq2seq任务上，而这正是LN的优势所在，由于LN是作用在单个训练数据的不同维度上，因此它能够在一条数据上进行normalization
  
4. 标签平滑归一化label smoothing regularization
通常我们使用交叉熵来计算预测误差时使用独热（one-hot）编码表示真实值，梯度下降算法为了减小误差会尽量是预测结果接近one-hot编码，也就是说，网络会驱使自身往正确标签和错误标签差值大的方向学习，在训练数据不足以表征所以的样本特征的情况下，这就会导致网络过拟合。
标签平滑归一化通过"软化"传统的独热one-hot类型编码，使得在计算误差值时能够有效抑制过拟合现象。它的实现非常简单，对于二值分类问题通过一个超参数$\epsilon$将原来的0，1分布变成$\epsilon, 1-\epsilon$分布，这样就可以缩短真假值之间的距离，最终起到抑制过拟合的效果。
5. warn-up learning rate
> If your data set is highly differentiated, you can suffer from a sort of "early over-fitting". If your shuffled data happens to include a cluster of related, strongly-featured observations, your model's initial training can skew badly toward those features -- or worse, toward incidental features that aren't truly related to the topic at all. Warm-up is a way to reduce the primacy effect of the early training examples. Without it, you may need to run a few extra epochs to get the convergence desired, as the model un-trains those early superstitions.
> Many models afford this as a command-line option. The learning rate is increased linearly over the warm-up period. If the target learning rate is  `p`  and the warm-up period is  `n`, then the first batch iteration uses  `1*p/n`  for its learning rate; the second uses  `2*p/n`, and so on: iteration  `i`  uses  `i*p/n`, until we hit the nominal rate at iteration  `n`.
> This means that the first iteration gets only 1/n of the primacy effect. This does a reasonable job of balancing that influence.
> Note that the ramp-up is commonly on the order of one epoch -- but is occasionally longer for particularly skewed data, or shorter for more homogeneous distributions. You may want to adjust, depending on how functionally extreme your batches can become when the shuffling algorithm is applied to the training set.


## Transformer的改进和发展
> ### Transformer 的局限性
> Transformer 无疑是对基于递归神经网络的 seq2seq 模型的巨大改进。但它也有自身的局限性：
> -   注意力只能处理固定长度的文本字符串。在输入系统之前，文本必须被分割成一定数量的段或块。
> -   这种文本块会导致**上下文碎片化**。例如，如果一个句子从中间分隔，那么大量的上下文就会丢失。
> 换言之，在不考虑句子或任何其他语义边界的情况下对文本进行分隔。
 那么，我们如何处理这些非常重要的问题呢？这就是使用过 Transformer 的人们提出的问题。由此催生了 Transformer-XL。
在这种架构中，在先前段中获得的隐状态被重用为当前段的信息员。它支持对长期依赖建模，因为信息可以从一个段流向下一个段。

- Transformer-XL

Transformer 架构可以学习长期依赖。但是，由于使用固定长度的上下文（输入文本段），它们无法扩展到特定的级别。为了克服这一缺点，这篇论文提出了一种新的架构：[《Transformer-XL：超出固定长度上下文的注意力语言模型》](https://arxiv.org/pdf/1901.02860.pdf)（Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context）

- 并行化
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
[Seq2seq pay Attention to Self Attention: Part 2](https://medium.com/@bgg/seq2seq-pay-attention-to-self-attention-part-2-cf81bf32c73d)
[Details Need More Attention: Transformer 没有被提到的细节](https://zhuanlan.zhihu.com/p/79987949)
[TRANSFORMERS FROM SCRATCH](http://www.peterbloem.nl/blog/transformers)
[Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwNTMwNzc0NzgsLTUzNTUyNDQxNCwxNj
kwMTEwOTMyLDEwNDU5MTg5MjEsMTcyODM0NzM1MSwxMTg2MDM5
MzcwLC03MDczMDQ0MzYsMTEyMTMyNjQzNywxNjAzNTE1OTI5LD
M2OTg3Njg4MCwtMTI5ODc0MTUwOCwtMTE1MzIzODMwOSwxNDMy
OTgyNzg1LDE5MTg2NDA4MzcsLTIxMDIwOTM5NjEsNzQzNDAwOD
E3LDIwMDU0NzkzMzIsMTg4Nzc0MDU4Miw5NzU2ODE0NDgsMTA5
NDc4NTkxNl19
-->