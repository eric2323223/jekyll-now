# Transformer-设计和构建高效的时序模型
在自然语言处理领域，循环神经网络RNN一直是被最广泛使用的深度机器学习模型，近年来卷积神经网络CNN也逐渐被引入用来提升训练效果。然而这两类模型都有一些难以克服的问题，Transformer模型以注意力机制为核心，并针对注意力机制的不足做了相关的设计和优化，取得了非常好的效果。本文我们就来一步步的分析和理解这个优秀的时序模型。

## 时序（seq2seq）问题
时序问题是应用机器学习（特别是深度学习）解决的一类常见问题，例如机器翻译，语态分析，摘要生成等自然语言处理问题（NLP）， 这类问题的最大特点是输入（或输出）以序列的形式出现，序列的长度可变，常见的NLP任务通常要求在分析整个输入序列的基础上才能产生输出。使用机器学习（深度学习）处理时序任务，通常使用编码器-解码器（encoder-decoder）架构，编码器负责将输入序列转换为包含整个序列所有特征的**序列编码**（context vector），解码器负责对这个内部表示进行解释。
![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vQpyCEO_5eiGEU2qG6G7ktzfhyjPRtMxtvGluMcFmeuEFoQYEMHIzAtvWAIH67v5uL1k5AKHS6Xn4cA/pub?w=680&h=255)

处理时序问题的传统方法是使用RNN模型，RNN能够保存状态，它将输入分为多步，依靠每步输入和上一步的状态更新当前的状态（和输出），通过重复这种步骤在读入所有序列元素后得到序列编码。由于RNN有存储机制并且不限制序列的长度，从模型结构上来说比较适合序列到序列问题。但是问题有三点
	  - 长序列的训练很困难，梯度下降算法在长序列的训练中容易发生梯度爆炸或梯度消失，虽然LSTM可以改善这个问题，但是在较长序列的训练中仍然无法完全避免。
	  - 只能顺序执行，无法通过并行加速训练
	  - 固定的存储空间在处理超长序列导致信息丢失？

为了解决RNN长序列训练问题，除了不断改进原生RNN之外，人们还尝试借助于CNN。这是由于CNN有能力处理一段输入序列而不是一个输入元素，虽然单个卷积核尺寸有限，可以通过堆叠多层卷积操作的方式逐步放大视域 。但这样做会不可避免的导致信息丢失（卷积操作中的上采样upsampling过程），同时增加了模型的复杂度。

上述两种模型对于长序列的处理都有缺陷，RNN需要一步一步的处理输入序列，CNN做出了一些改进但并不彻底。从根本上的解决长序列处理问题需要能一次性的处理全部输入（无论序列有多长），并且能根据这些输入信息分析序列元素之间的关联关系。人们从自己快速浏览的方式获得了启发，当人们需要快速浏览的时候不会按输入的顺序依次阅读，而会直接跳到需要关注的的部分，这种根据需要在不同位置跳跃的阅读方式和注意力相关，因此这种新的序列处理方式被命名为注意力机制。

## 注意力机制（attention mechanism）
基于组成整体的各个元素在整体中发挥的作用不相同这样一个事实，注意力机制的基本思想是在一定的目标下使用相对应的的权重组合各个序列元素来重新描述适合该目标的序列。这就好像在日常生活中，带着不同的目的看同一个事物会产生不同的理解。在下图中寻找生物会发现鱼和珊瑚，而寻找人工建筑则会发现钻井平台的支柱，正是由于不同的目标导致对图片的的物体分配了不同的权重，因此产生了不同的理解。

![enter image description here](https://www.capeandislands.org/sites/wcai/files/styles/medium/public/201609/oilrigs-5.jpg)
~~注意力机制主要用于seq2seq任务，它的基本思想就是对序列中的每个元素以一定的规则加入上下文信息。不同于RNN中先通过依次分析输入元素来逐步生成上下文CV的方式，注意力机制对这些输入元素进行加权平均的方式来一步加入所有元素信息来生成上下文context vector。这样做的好处是能够一步到位捕捉到全局的联系(序列元素直接进行两两比较),不仅大大加速（可以并行计算）了context vector的生成，而且避免了RNN的长序列训练困难的问题[^1]。~~

*[CV]: Context Vector

[^1]: the footnote
- 注意对象
- 注意规则
- 
从实现上来讲，注意力运算表现为加权求和运算，即对输入序列中的元素赋予相应的权重并相加。这里的权重来自任务目标，具体来说是根据目标对输出序列的要求，确定输出序列元素和输入序列元素之间的关系，再通过这种关系确定输入元素的权重。举个例子，对于机器翻译任务来说，由于我们需要让输入元素与输出元素表达相同的意义，因此需要比较它们的相似性，给相似性高的元素较高的权重，而对相似性低的元素赋予较低权重。

如果$X$表示输入序列集合$\{x_1, x_2, ... x_n\}$，$y$表示某个输出元素，$w_i$表示在对应$y$的计算过程中$x_i$的权重，可以将$X$对应$y$的注意力运算形式化的表示为
$$AttentionX_y=\sum_{i=1}^nw_ix_i$$
![enter image description here](http://www.peterbloem.nl/files/transformers/self-attention.svg)
如上所述，$w_i$决定于$x_i$和$y$的相关性$f(x_i,y)$，由于所有$x$都参与对应$y$的计算，所以使用softmax来保证所有权值之和等于1。
$$w_{i}=Softmax(Score(x_i,y))=\frac{exp(Score(x_i, y))}{\sum_{k=1}^nexp(Score(x_k, y))}$$
$f(x_i,y)$可以根据不同任务选择不同的计算方法，对于机器翻译任务来说，通常用矢量相似性来衡量元素的相关性，可以使用点积运算（dot product）
$$Score(x_i, y)=x_i\cdot y=|x_i||y|cos\theta=x_iy^T$$ 

> $\theta$表示两个向量$A,B$之间的夹角，如果$A,B$越相似则夹角$\theta$越小，$cos\theta$则越接近1
> 
> ![enter image description
> here](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSO0ZVpogoaP-ipyQF0Xhir4wSrgGJBdeU_5wDrea6UD9sF7icIYg)

上述表示对于目标为一个元素$y$时注意力计算的方法，大部分时序任务要求输出为序列，对于目标为序列的注意力计算方法是分别对每个元素$y_1, y_2, ... y_n$进行注意力计算，再将计算结果组成序列。
$$AttentionX_Y=\{AttentionX_{y_1}, AttentionX_{y_2}, ... AttentionX_{y_n}\}$$

从运算的结果上看，由于$AttentionX_{y_i}$包含了序列$X$所有元素的信息，因此我们也可以把注意力运算理解为**元素（$y_i$）在某一个序列上下文（$X$）环境中的重新定义**。这是一种对于时序任务非常有用的属性，RNN由于能够保存输入序列的信息而被广泛应用于时序任务，相比RNN通过逐步更新状态最终得到整个序列的信息的机制，注意力机制不但也有能力获取整个序列的信息，更重要的是它能一步直接得到结果，这使得注意力机制具备以下优势：
- 在并行方面，注意力机制不依赖于前一时刻的计算，可以很好的并行，优于RNN。
  传统方法使用RNN通过一步步的叠加分析过的输入来得到整个序列的内部表示（固定长度），Transformer模型中使用自注意力（self attention）机制来实现encoding，之所以称作自注意力是因为这是在输入序列内部进行的attention操作，由于attention操作就是对元素进行重新定义使其包含序列上下文信息，在输入序列元素进行attention的操作结果就是使该元素包含输入序列信息，因此经过self attention运算的整个输入序列的结果就是和一个输入序列大小一致的context vector。显然，self attention不需要想RNN那样一步步的出入输入，而是可以同时对每个元素进行attention运算，从下图可以发现，RNN需要在依次处理元素x1, x2和x3之后才能得到整个序列的上下文信息，而attention则可以同时处理x1，x2，x3而得到序列的上下文信息。![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vQZ5I4YZtpZOU8xnxqqJ2WVd7o9eeo0sHQa119cWm4qR85KanMs7-Z1DV1EfKxJLQrZaVglHLUJGPF2/pub?w=856&h=225)
- 在长距离依赖上，不管元素中间距离多远，路径长度总是1，可以轻松处理长距离依赖关系。RNN则存在梯度弥散或者梯度爆炸的问题。
- 注意力机制的计算复杂度更低，下表对注意力，RNN，CNN在计算复杂度上进行了对比，其中$length$表示序列长度，$dim$表示序列元素的维度，$kernel$表示卷积核的大小。由于在大部分自然语言处理任务中的元素维度都大于序列长度，因此对于这类任务来说注意力运算的计算复杂度要显著低于RNN和CNN。

  ||计算复杂度|
  |--|--|
  | 注意力 | $O(length^2 \cdot dim)$ |
  | RNN | $O(length \cdot dim^2)$ |
  | CNN | $O(length \cdot dim^2 \cdot kernel)$ |
 
注意力机制可以更一般的表示为
$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(Score(Q,K))V$$
这里的$K,V$分别表示一个键值对中的键key和值value，$Q$则表示注意目标query，这样我们之前的定义就变成$\mathrm{Attention}(Q, K, V)$在当$K=V$条件下的特殊形式。
下图左侧表示了$\mathrm{Attention}(Q, K, V)$的计算过程，按照由下向上的顺序:
1. 首先通过矩阵相乘MatMul运算得到$Q$和$K$的相似度，
2. 再通过Softmax操作转化为权重概率分布，
3. 最后使用MatMul加入$V$信息。

对于4维向量$K$=[Name, Age, Sex, Weight], 在4个维度上的取值分别为James, 25, male, 68kg，当以Age作为注意目标$Q$，以相似度作为注意力规则时。由于K的Age维度和Q的Age维度相似度很高，因此$score(K_{Age},Q_{Age})$接近于1（图中粉色和蓝色向量的方向接近），同理K的Name，Sex和Weight维度和Q的Age维度相似度接近于0（可以理解为粉色向量Q垂直于K1， K2和K3）。注意力计算的结果是4维向量，在Name，Sex，weight纬度上接近于0，该向量所包含的绝大部分信息都来自于Age维度，在该维度上的值为25。

![enter image description here](https://machinereads.files.wordpress.com/2018/09/scaled-dot-product-attention3.png?w=720)


## Transformer模型

Transformer来自于Google Brain团队2017年的文章Attention is all you need。正如论文的题目所述，整个网络结构完全是由注意力机制组成，由于没有使用RNN和CNN，避免了无法并行计算和长距离依赖等问题，用更少的计算资源，取得了更好的结果，刷新了多项机器翻译任务的记录。
整体架构上看，transformer仍属于编码器-解码器架构，通过编码器（encoder）将输入序列转换成内部表示，再通过不同解码器（decoder）实现不同的预测功能。~~从图中可以看到，编码器主要由两种组件构成：~~
![enter image description here](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/06/Screenshot-from-2019-06-17-20-01-32.png)

### 为什么Attention is all you need?
作为Transformer论文的最大创新，Transformer模型仅仅使用注意力机制不仅完成了以前需要RNN才能做到的工作，而且做的更快更好，下面我们就来看看Transformer是如何做到的。

#### 自注意力（self attention）
Transformer模型的首要工作就是使用编码器生成序列编码，前面我们介绍了注意力机制具备。。。的能力，在transformer的编码器中就是用了注意力机制来生成context vector，由于这种注意力机制的注意对象是输入序列自身，因此被称为自注意力。
时序问题（特别是自然语言处理问题）中的序列元素表示的含义通常不止该单个元素的的字面意义，而是与整个序列上下文有关系，因此在编码过程中需要考虑整个序列来决定其中每个元素的意义。自注意力机制中将每个元素都作为关注目标进行注意力计算，因此每个元素对每个元素都对在序列上下文中进行解释，很好的体现了这种通过全局确定局部的思想。
下图[^2]可视化的展示了在机器翻译任务下自注意力机制在对输入元素it的解释过程中，“the”和“animal”都发挥了比较大的权重。

![enter image description here](http://jalammar.github.io/images/t/transformer_self-attention_visualization.png)
[^2]: 来自Jay Alammar的著名博文The Illustrated Transformer

#### 注意力遮罩（Attention mask）
由于Attention可以直接看到所有的元素，因此需要一种手段来防止attention处理“不应该被看到的元素”，这是指在模型训练阶段不能让解码器的自注意力机制看到训练数据中当前时间点之后的正确预测值，否则模型就会利用标准答案“作弊”，如图所示。遮罩的实现很简单，即将被遮罩的元素设置为0。在Transformer的实现中除了解码器端的遮罩之外，还会在编码器-解码器注意力计算中，对。。。。。
- 编码器端自注意力用来生成context vector， 因此不需要遮罩
- 编码器-解码器注意力，需要对padding进行mask
- 解码器端自注意力，需要对当前位置之后的所有元素masking
  ![enter image description here](http://jalammar.github.io/images/gpt2/self-attention-and-masked-self-attention.png)

#### Scaled Dot-Product Attention (SDPA)
Transformer对标准的attention做了一个小小调整：加入特征缩放（feature scaling）。这样做主要是为了防止softmax运算将值较大的key过度放大，导致其他key的信息很难加入到attention结果中。
特征缩放体现在对$Q$和$K$计算点积$QK^T$以后，增加了一步除以$\sqrt{d_k}$运算。
$$\mathrm{SDPA}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

下图是上式的图像化表示，其中Scale就是特征缩放的操作。

>其中的权值来自该元素与其他元素的相似度，这是基于这样的假设-相似度越高的元素对确定该元素在整个序列中的含义的贡献度越大，由于序列元素以向量表示（word4vec），在transformer中使用点积运算来确定相似度，其结果是一个数值。形式化的定义为
$W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

![enter image description here](https://miro.medium.com/max/676/1*nCznYOY-QtWIm8Y4jyk2Kw.png)


### 位置编码（positional encoding）
与RNN和CNN不同，在注意力机制中没有先后顺序的概念（如第一个元素，第二个元素等）， 输入序列的所有元素都以没有特殊顺序或位置的方式输入网络，因此模型不知道元素的顺序。 因此，需要将与位置相关的信号添加到每个元素中，以帮助模型理解序列中元素的排列顺序。
一种最简单直接的位置编码方式是将每个元素的序号加入元素编码后再输入模型，这样做是否可行呢？ 考虑到序列的长度可以是任意长度，只讨论元素的绝对位置是不全面的（同一个词，在由3个词组成的句子中的第三个位置和30个词组成的句子中的第三个位置所表达的意思很可能是不一样的）。因此Transformer使用了基于周期函数（sin/cos函数）的位置编码方法。
位置编码$PE$可以表示为
$$PE_{{pos,2i}}=\sin(pos/10000^{2i/d_{model}}) $$
$$PE_{(pos, 2i+1)}=\cos(pos/10000^{2i/d_{model}})$$
其中$pos$表示位置，$i$表示元素编码的维度，$d_{model}$表示模型的维度，这种位置编码有如下优点：
- 利用sin/cos函数的周期性它能够进行任意长度序列的位置编码
- 由于sin(i+x)函数可以展开为sin(i)和cos(i)的线性表达式，使得$PE_{i+x}$的计算可以展开为$PE_i$的线性表达式，因此计算相对位置的效率比较高
- 使用多个不同频率来保证不会由于周期性导致不同位置的编码相同
- sin/cos函数的值总是在-1到1之间，这有利于神经网络的学习。

>![enter image description here](http://vandergoten.ai/img/attention_is_all_you_need/positional_embedding.png)

计算产生的位置编码是一个与元素具有相同维度的向量，使用相加的方式将位置信息叠加进元素中，如下图所示。作者没有在论文中解释为什么使用相加方式，直觉上来说相加会造成对元素向量的污染，而串联（concatenate）就不会有这种问题。一种解释是在高维中随机选择的向量几乎总是近似正交的，也就是说元素向量和位置编码向量是没有关联、相互独立的。因此尽管进行了矢量相加，但两个向量仍可以通过一些单个学习的变换而彼此独立地进行操作。也是正因为这种向量正交关系，串联并不会比相加表现得更好，但会大大增加学习参数方面的成本。
![enter image description here](https://wikidocs.net/images/page/31379/transformer6_final.PNG)


>为什么要同时使用sin和cos，而不只使用其中的一个？
下图可见
![enter image description here](https://i.stack.imgur.com/5QQmq.gif)
![enter image description here](https://i.stack.imgur.com/W0b0c.gif)


### 多头注意力（ Multiple Headed Attention, MHA)

Transformer仅仅使用注意力机制处理输入生成context vector，由于注意力机制本质上只是对输入进行加权平均运算，没有引入新参数也没有使用非线性运算，这导致特征提取能力不足，为了解决这个问题作者提出了多头注意力的方法。和卷积神经网络通过使用多个卷积核来发掘不同特征的思路类似，多头注意力也是通过多次随机初始化过程来提取不同特征。
下图中通过三次随机初始化分别得到了三种特征：红色表示动作，绿色表做动作施加者，蓝色表示动作承受着，可以看到在对“踢“进行了三次自注意力运算，分别对应三种特征。在对于动作信息的自注意力运算中，"我“和”球“的权值（灰色细线表示）比“踢”的权值（红色粗线）要小很多；同样，对动作施加者的自注意力运算中，“我”（绿色粗线）则是主要贡献者。在将三次自注意力运算的结果相加后，得到的新的“踢”的编码中就包含了三种特征的信息。现实中不可能每次随机初始化都能带来有效的特征，理论上随机初始化测次数越多就越有可能发现有效的特征，不过随之增长的是训练参数的增加，这意味着训练难度的提高，因此需要平衡，再Transformer模型中这个值是8。

![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vT4_Vn34rr1zN4OhXIo7oCGkzXDF__Y3CIVnZ_12fjqLHtKoRSJaVIyoR7ndQHtRlfNUmgecF5mucNg/pub?w=538&h=363)
具体实现来说是对同一个元素进行多次注意力运算， 每次注意力计算之前分别使用随机生成的参数$W^Q,W^K,W^V$通过矩阵相乘来初始化$Q,K,V$，
$$head_i =\mathrm{SDPA}(QW^Q_i, KW_i^K, VW_i^V)$$
- 对于编码器MHA，$Q, K, V$都是输入元素编码$x_i$
- 对于解码器MHA，$Q, K, V$都是已生成的输出元素编码$y_i$
- 对于编码器-解码器MHA， $Q$是输出元素编码$y_i$, $K,V$是context vector中的元素$c_i$

再将多次注意力运算的结果合并。合并的过程是首先对i次结果进行串联（concatenate），再通过和$W^O$进行矩阵相乘得到和输入同样维度的结果。
$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(head_i, ..., head_h)W^O$$

![enter image description here](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/img/MultiHead.png)


### Transformer全貌
在介绍了Transformer的主要组成部分之后，我们再来完整看一下Transformer模型。整体上来看，Transformer模型属于编码器-解码器架构，解码器需要根据序列编码sequence embedding（由编码器生成）和上一步的解码器输出来产生下一个输出，因此属于自回归(auto regressor)模型。
![enter image description here](https://camo.githubusercontent.com/4b80977ac0757d1d18eb7be4d0238e92673bfaba/68747470733a2f2f6c696c69616e77656e672e6769746875622e696f2f6c696c2d6c6f672f6173736574732f696d616765732f7472616e73666f726d65722e706e67)
Transformer的编码器和解码器分别有若干个编码层（解码层构成），每个编码层的结构完全一样，这些编码层相互串联在一起，编码器的输入首先进入第一个编码层，结算结果作为输入进入第二层，依次经过所有编码层后作为编码器的输出。
编码层由多头自注意力单元和按位前馈网络两部分组成。输入首先进入自注意力计算单元，再将计算结果输入按位前馈网络，这里的按位的含义是指每个位置的元素各自输入前馈网络里进行计算，前馈网络的结构为2个串联的全连接层，中间层维度较大（Transformer中为元素编码维度的4倍），最后一层的维度和元素编码的维度相同。这个设计的目的其实和多头注意力的设计类似，还是由于注意力机制在特征合成能力的不足，需要借助全连接网络的非线性计算来增加复杂特征合成的能力。
~~编码器由若干个（N）相同的编码层堆叠形成，每个编码层主要由一个多头注意力HMA和一个按位前馈网络构成，主要作用是将序列的上下文信息融入每个元素并进行特征合成。原始的输入编码首先经过位置编码器加入位置信息，在通过多个编码层生成包含位置信息，复杂特征信息的序列编码（context vector/sequence embedding）。~~
解码器的主要工作是根据context vector和上一步的输出计算下一步的输出。 它同样有多个结构相同的解码层串联而成，每个解码层由三部分组成。首先由解码器自多头注意力单元处理上一步的输出，计算后输入编码器-解码器多头注意力单元，编码器-加码器多头注意力单元还同时接收context vector，   接收两个输入，第一个输入是上一步的解码器输出（第一个解码器输出由一个固定的标识编码充当），这个输入进入~~位置编码加入位置信息，然后通过~~解码器的带遮罩的自注意力MHA（图中Masked Multi-Head Attention）加入上下文信息到已输出元素，处理完成后之后加入第二个输入context vector，通过进行编码器-解码器MHA加入来自编码器的特征信息，最后在经过按位前馈网络合成复杂特征。经过多个解码层处理后在通过全连接运算映射到目标词典空间，最后通过softmax选择可能性最大的元素作为输出。
工作流程：
1. 输入元素进行位置编码，位置编码与输入元素编码按位相加
2. 在编码层
	2.1 首先进行输入元素自注意力（多头注意力）计算，
	2.2 再将结果输入按位前馈网络
3. 重复多次编码层结算，结束编码阶段，得到context vector
4. 开始解码阶段，首先对输出元素进行位置编码（第一个输出为开始标记）, 输入元素与其位置编码按位相加
5. 在解码层
	7.1 首先进行输出元素（当前已输出）的多头自注意力计算
	7.2 进行编码（context vector）-解码（7.1结果）注意力计算
	7.3 对7.2结果输入按位前馈网络
8. 重复多次解码层计算
9. 通过全连接网络转化为目标词典宽度向量
10. 使用softmax确定输出元素（可能性最大）
11.  将当前输出元素输入6开始下一个输出元素的计算，直到输出为结束标记符
![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vSBNAHsyf_HP3_CkV1cygicnt0LhGxWcvw2PofecPP9TYJj41bghsAXTM6l6OSonSMvAjjgFInVDxC4/pub?w=961&h=590)
总结一下，attention是transformer的核心，它具有计算效率高（尤其对于长序列），可并行，容易训练等优势，但是同时也带了一些新问题：比如无序和特征合成能力下降。Transformer针对这些新问题分别提出了解决方案，如使用位置编码生成位置信息，使用多头注意力和按位前馈网络增强特征合成能力。

## Transformer优化技巧
由于Transformer属于比较复杂的深度模型，因此要通过使用一些优化技巧才能进行训练。Transformer中运用到的优化技术比较多，我们选择其中比较重要或者是有趣的来进行简单介绍
### 1. 残差链接(residual connection)
网络越深，表达能力越强，所以在需要表达复杂特征（如NLP，图像）的场景中使用的神经网络正在变得越来越深，但是深层网络带来了两个问题：1. 梯度弥散、爆炸，使得模型难以训练 2. 网络退化degradation，当网络深度到达一定后，性能不但不会随着深度的增加，反而会由性能下降。
![enter image description here](https://www.google.com/url?sa=i&source=images&cd=&ved=2ahUKEwjAjajGrMblAhXB26QKHZfDBS0QjRx6BAgBEAQ&url=https://www.researchgate.net/figure/A-cell-from-the-Residual-Network-architecture-The-identity-connection-helps-to-reduce_fig4_326786331&psig=AOvVaw1UDvQHXM-esMFq1rcNP7FV&ust=1572606118049027)
残差链接用一个简单的办法巧妙的解决了这两个问题，就是将两个不相邻网络层直接连接（短接）。这样梯度gradient可以跨越中间层直接传递，避免经过中间层时梯度被多次缩放导致梯度弥散（爆炸）的问题；另一方面，实验证明当使用RELU作为激活函数时，残差连接也以有效防止网络退化。原因。。。
在transformer中的每一个编码层（解码层）都使用了残差连接来分别短接多头注意力和按位前馈网络，这样做一来解决了梯度问题，同时还能帮助位置信息顺利传递到高层去
### 2. Layer normalization
  Normalization是在机器学习中常用的一种数据预处理方法，为了更有效的运行机器学习算法，需要将原始数据“白化”Whitening，也就是在统计学中常常提到的使数据“独立，同分布”。
   目前在深度学习中最常用的是BN，它是对不同训练数据的同一维度进行normalization，这种方法可以有效缓解深度模型训练中的*梯度爆炸、弥散的问题*。而在transformer采用了相对冷门的LN，主要原因是BN很难应用在训练数据长度不同的seq2seq任务上，而这正是LN的优势所在，由于LN是作用在单个训练数据的不同维度上，因此它能够在一条数据上进行normalization
  
### 3. 标签平滑归一化label smoothing regularization
通常我们使用交叉熵来计算预测误差时使用独热（one-hot）编码表示真实值，梯度下降算法为了减小误差会尽量使预测结果接近one-hot编码，也就是说，网络会驱使自身往正确标签和错误标签差值大的方向学习，在训练数据不足以表征所以的样本特征的情况下，预测结果的置信度过高会导致网络过拟合。
标签平滑归一化通过"软化"传统的独热编码，使得训练时能够有效抑制过拟合现象。它的实现非常简单，通过一个超参数$\epsilon \in(0,1)$将原来的0，1分布变成$\epsilon, 1-\epsilon$分布（对于二值分类问题），这样就缩短了真假值之间的距离，最终起到抑制过拟合的效果。
### 4. 学习率热身Learning rate warm up
 训练初期由于离目标较远，一般需要选择大的学习率，但如果训练数据集具有高度的差异性则使用过大的学习率可能导致不稳定性。这是由于如果初始化后的数据恰好只包含一部分特征，则模型的初始训练可能会严重偏向于这些特征，这会增加模型学习其他特征的难度。
 所以可以做一个学习率热身阶段，在开始的时候先使用一个较小的学习率，然后当训练过程稳定的时候再把学习率调回去。在预热期间，学习率呈线性增加。如果目标学习率是$p$，预热期是$n$，则第一批迭代将$p/n$用作学习率；第二个使用$2*p/n$，依此类推：迭代$i$使用$i*p/n$，直到我们在迭代$n$次后达到学习率$p$。

## Transformer的改进和发展
Transformer取得巨大成功引起关注，学术和产业界都在尝试在实现和理论层面对他进行改进
### Transformer-XL
虽然理论上Transformer可以处理任意长度的输入，但在实际的运用中资源是有限的，因此Transformers目前使用固定长度的上下文来实现，即将一个长的文本序列截断为几百个字符的固定长度片段，然后分别处理每个片段。这种操作会使相邻块片段之间的上下文丢失  ，导致上下文碎片化。Transformer-XL基于以下两种关键技术解决了这个问题：
	- 片段级递归机制(segment-level recurrence mechanism) 
	主要解决上下文碎片化问题，使上下文信息现在可以跨片段边界流动。思路是将上一片段segment的memory传到下一片段的同样位置
	![enter image description here](https://miro.medium.com/max/2152/1*Y3rxi7H06Ir-q_W2Q2zSIg.png)
	- 相对位置编码方案(relative positional encoding scheme)。
	由于transformer上的位置编码方案会导致不同块的元素具有相同的位置编码，因此提出了一种新的位置编码，它是每个attention模块的一部分，基于元素之间的相对距离而不是它们的绝对位置。

### 并行化
Despite not having any explicit recurrency, implicitly the model is built as an autoregressive one. It implies that in order to generate an output (both while training or during inference), the model needs to compute previous outputs, which is extremely costly, for the whole net has to be run for every output. That’s the main idea to overcome in a recent paper by researchers at [_Salesforce Research_](https://einstein.ai/research/non-autoregressive-neural-machine-translation) and the University of Hong Kong, who tried to make the whole process parallelizable[23](https://ricardokleinklein.github.io/2017/11/16/Attention-is-all-you-need.html#fn:23). Their proposal is to compute _fertilities_ for every input word in the sequence, and use it instead of previous outputs in order to compute the current output. This is summarized in the figure below.
尽管没有任何显式递归，但是隐式地将模型构建为自回归模型。 这意味着为了生成输出（在训练时或在推理期间），该模型需要计算先前的输出，这非常昂贵，因为必须为每个输出运行整个网络。 这是Salesforce Research和香港大学的研究人员在最近的一篇论文中要克服的主要思想，他们试图使整个过程可并行化。 他们的建议是为序列中的每个输入单词计算肥力，并使用它代替先前的输出以计算当前输出。 下图对此进行了总结。
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
[When Does Label Smoothing Help?](https://medium.com/@nainaakash012/when-does-label-smoothing-help-89654ec75326)
[Attention Is All You Need](https://machinereads.com/2018/09/26/attention-is-all-you-need/)
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjYwNTE1OTc3LC0xNjUwMjM2NjcsMTY5OD
Q5NDY2MCw5NzY4MjU3OTAsLTEwOTQ5ODQwOTgsMTIwMTc2MDQ4
Niw1MDE3MzMwMjgsODM2ODEyMjQxLDEzNzM4MTkxMjYsMTYxND
Q2NTE0NSwtMzY4NTUwODU5LC0xMTYzODI3NjExLC0xNDA3MjUx
NzU0LDE5Njk0NTk2MTYsMTU5NjQ0MDU0MCw5NjA3MTAzMzYsLT
c1NTc0ODMzOCwtNDI4Mzc1MDQwLDE2OTM0MzUyMTUsMTEyMDA5
Nzk2Ml19
-->