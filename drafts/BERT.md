# NLP的迁移学习-BERT篇
迁移学习让普通人应用复杂强大模型解决实际问题的捷径，利用注意力机制的强大能力，BERT在NLP领域的一系列任务的基准测试中取得了新高。本文旨在介绍BERT的结构，特性，预训练方法和微调方法，并试图解释BERT模型设计背后的原因。最后回归应用，介绍了如何利用BERT预训练模型在colab平台快速实现智能问答。
1. 迁移学习和预训练模型
    1.1 NLP的迁移学习
    1.2 语言模型
2. BERT简介
3. BERT模型结构
    3.1 编码层
    3.2 Transformer编码器
4. BERT的预训练
    4.1 任务设计
    4.2 预训练流程
    4.3 优化
5. BERT的微调
    5.1 情绪分析任务
    5.2 名称实体识别NER任务
    5.3 通用语言理解GLUE任务
    5.4 问答SQuAD任务
6. BERT的应用-实现一个智能问答机器人
    6.1 环境搭建
    6.2 实验流程
7. 总结


self-supervised learning is important area because it can greatly reduce the effort of training deep model, 

作为NLP迁移学习的成功应用，BERT证明了。。。本文旨在介绍BERT模型的结构和设计原理，以及BERT的应用。
## 迁移学习和预训练模型
![enter image description here](https://miro.medium.com/max/3283/1*Z11P-CjNYWBofEbmGQrptA.png)
迁移学习旨在通过重用 。。。来加速学习和增强预测的准确性，对于当今越来越复杂的神经网络来说，需要巨大的人力物力和时间成本。。。使用迁移学习是非常有意义的。通过再imagenet训练视觉特征提取网络，数据比较从头训练和使用迁移训练。。。
现实的问题是获取足够的标记数据非常困难，因此
### NLP的迁移学习
我们知道在CV中的迁移学习过程是首先训练一个通用的的图像特征提取模型（如VGG19， ResNet50等），再结合下游任务需要通过扩展第一阶段的模型来进行fine tuning。进行与CV任务类似，应用迁移学习解决NLP问题也可以分为两个阶段。首先通过预训练学习出可重用的特征提取模型，也叫预训练模型。
> NLP的最大挑战之一是缺乏足够的培训数据。总体而言，有大量文本数据可用，但是如果我们要创建特定于任务的数据集，则需要将该堆划分为很多不同的字段。而当我们这样做时，我们最终仅得到数千或数十万个人标记的培训示例。不幸的是，为了表现良好，基于深度学习的NLP模型需要大量的数据-在数百万或数十亿的带注释的训练示例上进行训练时，他们看到了重大改进。为了帮助弥合数据鸿沟，研究人员开发了各种技术，可在网络上使用大量未注释的文本来训练通用语言表示模型（这称为预训练）。然后，可以在较小的特定于任务的数据集上微调这些通用的预训练模型，例如，在处理诸如问题回答和情感分析之类的问题时。与从头开始对较小的特定于任务的数据集进行训练相比，此方法可显着提高准确性。

![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vStoAwye3EraSC6HH5m_S8VOsVEp3hsTtQuAVF-dEmPlFvEZqAxBHDQryl3FnVf_BZ6Csb969AGbChe/pub?w=791&h=385)

由于NLP主要关注语言（字符序列）的理解和处理，作为语言基本组成单位的词（word）也就自然成为了预训练的关注点。预训练的目标经历逐步的发展变化
#### 预训练 pre training

- output: embeddings
	- 静态词编码（static word embedding），比Word2Vec，Glove等，顾名思义这类编码赋予每个词固定的编码值，并且编码值体现了词的代表的含义，我们可以通过对编码值的运算得到有意义的结果，比如著名的例子
***king — man + woman = queen***
![enter image description here](https://miro.medium.com/max/634/1*dm9dudL37B6JG8saeR3zIw.png)

	- 语境词编码（contextualized word embedding），静态词编码的最大的问题在于它只能个每一个词一个编码值，无法处理一词多义的情况。将“我爱吃苹果”和“我爱苹果手机”中的苹果赋予相同的编码是不合适的，更合理的方式是通过结合词出现的上下文判断词的含义，比如通过“吃”和“手机”来判断上面两句话中的“苹果”分别代表一种水果和一个品牌，这就是语境词编码的基本思想。所以从使用者角度来说，我们需要一个模型能过通过输入语句得到（计算出）该语句的含义，或者该语句中每个词的含义。从这个意义上讲，我们本质上需要的是一种能够提取语义特征的能力，这和CV中的迁移学习的目标是一致的。
		
- self-supervised learning
Imagenet将超过一千四百万图片通过众包的方式进行人工标注，将他们分成2万多个不同分类，这项从2007年开始的浩大工程为计算机视觉图形相关的监督式机器学习提供了高质量的训练数据，从而为CV迁移学习打下了基础。同样的为了NLP领域也由类似的需求：为每个词建立正确的标签数据来帮助进行监督训练，根据语言的特点，设计了语言模型（Language Model）这种训练任务来进行。。。LM属于自监督（self supervised）训练方法，使用这种训练方法不需要为语句进行人工标注，而只使用语句序列本身就可以进行训练。LM是一种统计方法，用于计算一个序列$W$（由词$w_i, w_2, ... w_m$组成的一句话）出现的概率$$P(W)=P(w_1,w_2,w_3,...w_m)$$LM也可以用于计算在一个序列中某个词$w_{n+1}$出现的概率$$P(w_{n+1}|w_1,w_2, w_3,...w_n)$$
根据这样一个基本假设：正确的语句出现的概率比不正确的语句出现的概率大
The good LM should calculate higher probabilities to “real” and “frequently observed” sentences than the ones that are wrong accordingly to natural language grammar or those that are rarely observed.
-   **Machine translation:**  translating a sentence saying about height it would probably state that  P(tall  man)>P(large  man)P(tall man)>P(large man)  as the ‘_large_’ might also refer to weight or general appearance thus, not as probable as ‘_tall_’
    
-   **Spelling Correction:**  Spell correcting sentence: “Put you name into form”, so that  P(name  into  form)>P(name  into  from)
由此我们选择概率最大的词作为预测值$$\argmax P(w_n|w_1,w_2,w_3,...w_{n-1})$$
	- 使用LM进行训练，可以按照从前到后的顺序进行预测，比如通过“”判断后一个词是“”，也可以按照从后向前的顺序，$$\argmax P(w_i|w_n,w_{n-1},w_{n-2}, ...w_{i+1})$$比如通过“”判断前一个词是“”。
>  今天 天气 不错， 我们 去 公园 玩 吧。

这句话，单向语言模型在学习的时候是从左向右进行学习的，先给模型看到“今天 天气”两个词，然后告诉模型下一个要填的词是“不错”。然而单向语言模型有一个欠缺，就是模型学习的时候总是按照句子的一个方向去学的，因此模型学习每个词的时候只看到了上文，并没有看到下文。更加合理的方式应该是让模型同时通过上下文去学习，这个过程有点类似于完形填空题。例如：

>今天 天气 { }， 我们 去 公园 玩 吧。

通过这样的学习，模型能够更好地把握“不错”这个词所出现的上下文语境。

#### 微调 fine tune
由于使用海量的数据进行预训练，预训练模型通常具有一般的常识，由此作为基础再进行微调，使得模型能更好的适合特定任务。
- 模型调整
通常做法是在预训练模型基础上增加任务相关的层，如由全连接层和softmax运算构成的分类层用于分类任务。
- supervised learning
使用少量任务相关的标记数据来进行微调，通常的做法是在预训练模型的后面直接加上上一个分类器（由全连接和softmax运算构成）使模型输出一个预测类型，计算cross entropy误差从而通过反向传递更新模型参数。
	- 更新全部模型参数
	- 只更新任务层参数 - 预训练模型只作为特征提取器


## ~~- unsupervised fine tuning? - clustering and measure class separation - classify result by compute distances to different classes -~~


- zero shot learning
~~无微调适用于容量更大预训练模型，这类模型一般包含了更多的常识，比如GPT2使用了xx的高质量数据进行预训练，无需微调也可能在不同下游任务重生成可接受的预测。对于这类模型，只需要给出少量的样例让模型理解预测意图。。。~~

## BERT简介
BERT（Bidirectional Encoder Representations from Transformer）是一个用于提取输入序列特征信息的预训练模型。When BERT was published it achieved [state-of-the-art] performance in 11 [natural language understanding] tasks:[[1]] [GLUE]task set (consisting of 8 tasks), [MultiNLI] [SQuAD] v1.1, SQuAD v2.0
2018, google发表了论文BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding， 2019年google将BERT模型应用到了搜索服务中，现在已经支持了超过70种语言

BERT最大的创新是将Transformer模型应用到了语言模型中，。。。。影响和决定了BERT很多特殊性质。在BERT之前，

- context dependent embedding
BERT模型生成的元素编码属于动态编码，它能根据输入序列生成每个序列元素（word）在序列上下文中的特征向量
- bidirectional Language Model
这是由于它是以Attention机制为基础。注意力机制可以一次看到所有的序列元素，每个元素的编码的计算都包含了该元素之前和之后的序列信息，因此BERT属于双向语言模型，并且由于能够同时看到前向和后向的信息，BERT不同于以往的双向语言模型，如ELMO，。。。。。。deep bidirectional 
并非所有的基于attention机制的模型都是双向语言模型，比如GPT使用了遮罩的方式使模型无法看到当前元素之后的序列信息，因此它属于单向语言模型。

- 

## BERT模型结构
### Transformer encoder based
BERT模型主要包含这个部分，编码层和Transformer编码器
![enter image description here](https://www.lyrn.ai/wp-content/uploads/2018/11/transformer.png)

### 编码层
编码层的作用是
1. 将输入语句（BERT is powerful）转换为模型可处理的浮点数向量
2. 加入特殊符号[CLS][SEP] -- No! this is done in data preprocessing

    embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    为什么可以相加？[https://www.zhihu.com/question/374835153/answer/1069173198](https://www.zhihu.com/question/374835153/answer/1069173198)

[https://mc.ai/why-bert-has-3-embedding-layers-and-their-implementation-details/](https://mc.ai/why-bert-has-3-embedding-layers-and-their-implementation-details/)
![enter image description here](https://i.stack.imgur.com/QCcYF.png)
- 词编码(config.vocab_size, config.hidden_size, padding_idx=0)
[https://www.topbots.com/generalized-language-models-bert-openai-gpt2/#input-embedding](https://www.topbots.com/generalized-language-models-bert-openai-gpt2/#input-embedding)
- 段编码(config.type_vocab_size, config.hidden_size)
在BERT处理多条语句时，用于区分不同语句
- 位置编码(config.max_position_embeddings, config.hidden_size)
由于注意力计算不关心输入序列元素的先后循序，因此需要事先加入位置信息再输入模型。不同于Transformer的基于周期函数的固定位置编码方法，BERT采用可学习的位置编码方式，bert中的最大句子长度是512 所以Position Embedding layer 是一个size为（512，768）的lookup table，其中的每一个元素都是可学习的参数，随预训练这些位置相关的参数收敛，。。。**相比Transformer的位置编码，似乎没考虑相对位置????**
### Transformer编码器
Transformer模型是由google ai于2017年发布的一个编码器-解码器架构模型，最初应用于机器翻译。Transformer的最大特点是使用注意力机制（attention mechanism），解决了使用RNN模型造成的梯度爆炸和无法并行的问题，并且实践证明transformer中提出的多头注意力具有强大的特征提取能力，性能超越了RNN,CNN等传统方法。
> Transformer所使用的注意力机制的核心思想是去计算一句话中的每个词对于这句话中所有词的相互关系，然后认为这些词与词之间的相互关系在一定程度上反应了这句话中不同词之间的关联性以及重要程度。因此再利用这些相互关系来调整每个词的重要性（权重）就可以获得每个词新的表达。这个新的表征不但蕴含了该词本身，还蕴含了其他词与这个词的关系，因此和单纯的词向量相比是一个更加全局的表达。
> Transformer通过对输入的文本不断进行这样的注意力机制层和普通的非线性层交叠来得到最终的文本表达。

Transformer由编码器和解码器组成，编码器负责将输入序列中的每个元素（word）转换为包含上下文信息的特征向量，再由解码器根据编码后的特征向量生成输出序列。BERT模型中只使用了transformer的编码器，它主要由若干个结构相同的编码层连接而成。每一个编码层主要有一个多头自注意力计算单元（Multi-Head Attention）和按位前馈网络(Feed Forward)组成，多头自注意力计算单元负责为每个输入元素生成特征向量，前馈网络能够通过组合元素特征向量生成更复杂的特征向量。

![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vSqp25HORnsDrfUfkTFUgKeTC7IITVZrTMXBuf6eSp4_HmCsGRoGwAxEoN87fuhT98Xsc4IulE_U4vM/pub?w=960&h=720)
## BERT的预训练
### 任务设计
BERT的预训练被设计为多任务学习（multi-task learning），包含两个任务：
- MLM
[https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
给定一个句子，会随机Mask 15%的词，然后让BERT来预测这些Mask的词，如同上述10.1所述，在输入侧引入[Mask]标记，会导致预训练阶段和Fine-tuning阶段不一致的问题，因此在论文中为了缓解这一问题，采取了如下措施：

如果某个Token在被选中的15%个Token里，则按照下面的方式随机的执行：

-   80%的概率替换成[MASK]，比如my dog is hairy → my dog is [MASK]
-   10%的概率替换成随机的一个词，比如my dog is hairy → my dog is apple
-   10%的概率替换成它本身，比如my dog is hairy → my dog is hairy

这样做的好处是，BERT并不知道[MASK]替换的是这15%个Token中的哪一个词(**注意：这里意思是输入的时候不知道[MASK]替换的是哪一个词，但是输出还是知道要预测哪个词的**)，而且任何一个词都有可能是被替换掉的，比如它看到的apple可能是被替换的词。这样强迫模型在编码当前时刻的时候不能太依赖于当前的词，而要考虑它的上下文，甚至对其上下文进行”纠错”。比如上面的例子模型在编码apple是根据上下文my dog is应该把apple(部分)编码成hairy的语义而不是apple的语义。
细节三：对于任务一，对于在数据中随机选择 15% 的标记，其中80%被换位[mask]，10%不变、10%随机替换其他单词，原因是什么？

**两个缺点：**

1、因为Bert用于下游任务微调时， [MASK] 标记不会出现，它只出现在预训练任务中。这就造成了预训练和微调之间的不匹配，微调不出现[MASK]这个标记，模型好像就没有了着力点、不知从哪入手。所以只将80%的替换为[mask]，但这也**只是缓解、不能解决**。

2、相较于传统语言模型，Bert的每批次训练数据中只有 15% 的标记被预测，这导致模型需要更多的训练步骤来收敛。
- NSP
### 损失函数
total_loss = masked_lm_loss + next_sentence_loss
BERT的损失函数由两部分组成，第一部分是来自 Mask-LM 的**单词级别分类任务**，另一部分是**句子级别的分类任务**。通过这两个任务的联合学习，可以使得 BERT 学习到的表征既有 token 级别信息，同时也包含了句子级别的语义信息。具体损失函数如下：

![[公式]](https://www.zhihu.com/equation?tex=L%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B1%7D%2C+%5Ctheta_%7B2%7D%5Cright%29%3DL_%7B1%7D%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B1%7D%5Cright%29%2BL_%7B2%7D%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B2%7D%5Cright%29)

其中  ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta)  ​ 是 BERT 中 Encoder 部分的参数，​  ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_1)  是 Mask-LM 任务中在 Encoder 上所接的输出层中的参数，​  ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_2)  则是句子预测任务中在 Encoder 接上的分类器参数。因此，在第一部分的损失函数中，如果被 mask 的词集合为 M，因为它是一个词典大小 |V| 上的多分类问题，那么具体说来有：

![[公式]](https://www.zhihu.com/equation?tex=L_%7B1%7D%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B1%7D%5Cright%29%3D-%5Csum_%7Bi%3D1%7D%5E%7BM%7D+%5Clog+p%5Cleft%28m%3Dm_%7Bi%7D+%7C+%5Ctheta%2C+%5Ctheta_%7B1%7D%5Cright%29%2C+m_%7Bi%7D+%5Cin%5B1%2C2%2C+%5Cldots%2C%7CV%7C%5D)

在句子预测任务中，也是一个分类问题的损失函数：

![[公式]](https://www.zhihu.com/equation?tex=L_%7B2%7D%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B2%7D%5Cright%29%3D-%5Csum_%7Bj%3D1%7D%5E%7BN%7D+%5Clog+p%5Cleft%28n%3Dn_%7Bi%7D+%7C+%5Ctheta%2C+%5Ctheta_%7B2%7D%5Cright%29%2C+n_%7Bi%7D+%5Cin%5B%5Ctext+%7BIsNext%7D%2C+%5Ctext+%7BNotNext%7D%5D)

因此，两个任务联合学习的损失函数是：

![[公式]](https://www.zhihu.com/equation?tex=L%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B1%7D%2C+%5Ctheta_%7B2%7D%5Cright%29%3D-%5Csum_%7Bi%3D1%7D%5E%7BM%7D+%5Clog+p%5Cleft%28m%3Dm_%7Bi%7D+%7C+%5Ctheta%2C+%5Ctheta_%7B1%7D%5Cright%29-%5Csum_%7Bj%3D1%7D%5E%7BN%7D+%5Clog+p%5Cleft%28n%3Dn_%7Bi%7D+%7C+%5Ctheta%2C+%5Ctheta_%7B2%7D%5Cright%29)

具体的预训练工程实现细节方面，BERT 还利用了一系列策略，使得模型更易于训练，比如对于学习率的 warm-up 策略，使用的激活函数不再是普通的 ReLu，而是 GeLu，也使用了 dropout 等常见的训练技巧。
### 预训练流程
[http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)  Recapping a sentence’s journey
1. Preprocessing: Add special token to raw input: "BERT is awesome. BERT is wonderful" becomes "[CLS] BERT is awesome [SEP] BERT is wonderful [SEP]"
2. Embedding
	2.1 word embedding: tokenization
	2.2 positional embedding
	2.3 segment embedding
3. Transformer encoder: 

like this



## BERT的微调fine tune

### 微调任务类型
![enter image description here](https://lilianweng.github.io/lil-log/assets/images/BERT-downstream-tasks.png)
### **7.1 针对句子语义相似度的任务**
  
![](https://pic1.zhimg.com/80/v2-971f887ed616ea0f65941c8dc15ee128_720w.jpg)

  实际操作时，上述最后一句话之后还会加一个[SEP] token，语义相似度任务将两个句子按照上述方式输入即可，之后与论文中的分类任务一样，将[CLS] token位置对应的输出，接上softmax做分类即可(实际上GLUE任务中就有很多语义相似度的数据集)。

### **7.2 针对多标签分类的任务**

多标签分类任务，即MultiLabel，指的是一个样本可能同时属于多个类，即有多个标签。以商品为例，一件L尺寸的棉服，则该样本就有至少两个标签——型号：L，类型：冬装。

对于多标签分类任务，显而易见的朴素做法就是不管样本属于几个类，就给它训练几个分类模型即可，然后再一一判断在该类别中，其属于那个子类别，但是这样做未免太暴力了，而多标签分类任务，其实是可以**只用一个模型**来解决的。

利用BERT模型解决多标签分类问题时，其输入与普通单标签分类问题一致，得到其embedding表示之后(也就是BERT输出层的embedding)，有几个label就连接到几个全连接层(也可以称为projection layer)，然后再分别接上softmax分类层，这样的话会得到​  ![[公式]](https://www.zhihu.com/equation?tex=loss_1%2C%5C+loss_2%2C%5C+%5Ccdots%2C%5C+loss_n)  ，最后再将所有的loss相加起来即可。这种做法就相当于将n个分类模型的特征提取层参数共享，得到一个共享的表示(其维度可以视任务而定，由于是多标签分类任务，因此其维度可以适当增大一些)，最后再做多标签分类任务。

### **7.4 文本生成？NO!**

### 微调技巧
1. 调整参数（内存），模型选择
2.  **长文本处理**
[https://zhuanlan.zhihu.com/p/109143667](https://zhuanlan.zhihu.com/p/109143667)
	对于长文本文中做了两种处理方式，截断和切分。

	-   截断：一般来说文本中最重要的信息是开始和结尾，因此文中对于长文本做了截断处理。

	> head-only：保留前510个字符  
	> tail-only：保留后510个字符  
	> head+tail：保留前128个和后382个字符
	
	- 切分: 将文本分成k段，每段的输入和Bert常规输入相同，第一个字符是[CLS]表示这段的加权信息。文中使用了Max-pooling, Average pooling和self-attention结合这些片段的表示。
	- 
下面是实验的结果，head+tail的表示在两个数据集上的效果都比较好。应该是长文本结合了句首和句尾的信息，获取的信息比较均衡。不过奇怪的是拼接的方式整体居然不如截断，个人猜测可能是将句子切成几段之后增加了模型的不稳定性，而错误叠加起来可能就会被放大。而max-pooling和self-attention也更加强调了文本中比较有用的信息，所以整体效果优于average.
![enter image description here](https://pic3.zhimg.com/80/v2-f932b2ed7aa4af745b512e2e0f43093e_720w.jpg)

## BERT的改进
[关于BERT的若干问题整理记录](https://zhuanlan.zhihu.com/p/95594311)
### task design
- spanBERT [https://zhuanlan.zhihu.com/p/75893972](https://zhuanlan.zhihu.com/p/75893972)
### distillation
### LAMP？not a BERT improvement
## BERT应用
[https://github.com/ProHiryu/bert-chinese-ner](https://github.com/ProHiryu/bert-chinese-ner)
[https://github.com/chiahsuan156/ODSQA](https://github.com/chiahsuan156/ODSQA)
### environment-colab
 - User BERT base model
 - Tweak: batch size, max length
 - Mixed precision training
 - Gradient checkpoint
### huggingface transformer
### BERT as a service
### DistilBERT
### SQUAD

## 总结













	
## Transfer learning
- what?
- why?
	- Deep model which has lots of parameters need lots of training data
	- labeled training data is very expensive
	- To save training efforts (save money and time)
	- Transfer learning has been successful in CV tasks
- how?
	- pretraining - generate embeddings (word embeddings)
	- supervised finetuning - train for downstream tasks
	
### sequential transfer learning
- pretraining
- Adaptation
- finetuning
	- supervised
	- unsupervised

## pretraining
- self-supervised learning自监督学习 based on  Language Model
	- Many successful pretraining approaches are based on language modeling
	- Informally, a LM learns Pϴ(text) or Pϴ(text | some other text)
	- Doesn’t require human annotation
	- Many languages have enough text to learn high capacity model
	- Versatile—can learn both sentence and word representations with a variety of
objective functions
- from static embedding to dynamic embedding
	- static encoding (contextless embedding)- word2vec, glove
	- dynamic encoding (contextual embedding) - elmo, bert
- How LM help NLP transfer learning
	- feature based
	**Feature-based**指利用语言模型的中间结果也就是LM embedding, 将其作为额外的特征，引入到原任务的模型中。通常feature-based方法包括两步：

		1.  首先在大的语料A上无监督地训练语言模型，训练完毕得到语言模型
		2.  然后构造task-specific model例如序列标注模型，采用有标记的语料B来有监督地训练task-sepcific model，将语言模型的参数固定，语料B的训练数据经过语言模型得到LM embedding，作为task-specific model的额外特征。ELMo是这方面的典型工作，请参考[2]
	- fine-tuning
	Fine-tuning方式是指在已经训练好的语言模型的基础上，加入少量的task-specific parameters, 例如对于分类问题在语言模型基础上加一层softmax网络，然后在新的语料上重新训练来进行fine-tune。例如OpenAI GPT [3] 中采用了这样的方法，模型如下所示

![](https://pic1.zhimg.com/80/v2-8f857288cf73acba9ddb6b3742265144_hd.jpg)

图2 Transformer LM + fine-tuning模型示意图

  
首先语言模型采用了Transformer Decoder的方法来进行训练，采用文本预测作为语言模型训练任务，训练完毕之后，加一层Linear Project来完成分类/相似度计算等NLP任务。因此总结来说，LM + Fine-Tuning的方法工作包括两步：

1.  构造语言模型，采用大的语料A来训练语言模型
2.  在语言模型基础上增加少量神经网络层来完成specific task例如序列标注、分类等，然后采用有标记的语料B来有监督地训练模型，这个过程中语言模型的参数并不固定，依然是trainable variables.
- Encoding
	- character level
	- BPE
	- word level
- task design (training objective) for self-supervised learning
	- Language model
	- bidirectional LM
	- MLM, NSP
	- GAN (ELATRA)
- The pretrained model is too complex to use
	- distillation
	- 
## Adaptation
GPT-2论证了什么事情呢？对于语言模型来说，不同领域的文本相当于一个独立的task，而如果把这些task组合起来学习，那么就是multi-task学习。所特殊的是这些task都是同质的，即它们的目标函数都是一样的，所以可以统一学习。那么当增大数据集后，相当于模型在更多领域上进行了学习，即模型的泛化能力有了进一步的增强。
### GPT-2 直接做下游任务

除了语言模型上的进展之外，GPT-2还首次尝试了直接用语言模型做下游任务，也就是不用在具体任务上的损失函数。这是如何做到的呢？

比如，如果是summarization任务，那么对于语言模型来说，我加一个新词TL;DR:, 改词前面是context，后面是摘要。那么语言模型遇到这个词后，就能推断出来，接下来要做抽摘要的工作了。

同理，对于translate任务，我们把数据做成 french sentence = english sentence，那么语言模型遇到=的时候，应该能推断出接下来是翻译任务。

虽然在这些任务上，GPT-2都没有达到SOTA的效果，但是效果也是相当可观的。表明了高容量模型在这个方向上的可能性。

## Downstream fine-tuning
- finetuning tips - ULMFit
- tools: TF-hub

### Example

[Generalized language model](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)
[How to build openai's GPT2](https://blog.floydhub.com/gpt2/)
[BERT Explained: State of the art language model for NLP](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
[NLP预训练演进 - from Word2Vec to XLNet](https://zhuanlan.zhihu.com/p/93343298)
[nlp中的词向量对比：](https://zhuanlan.zhihu.com/p/56382372)
[史上最全词向量讲解](https://zhuanlan.zhihu.com/p/75391062)
[ELECTRA: 超越BERT, 19年最佳NLP预训练模](https://zhuanlan.zhihu.com/p/89763176)
[从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)
[如何评价BERT-回答](https://www.zhihu.com/question/298203515/answer/516170825)
[NLP规则改写](https://zhuanlan.zhihu.com/p/47488095)
[BERT Explained: A Complete Guide with Theory and Tutorial](https://towardsml.com/2019/09/17/bert-explained-a-complete-guide-with-theory-and-tutorial/)
[BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
[Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)
[ELMO](https://petrlorenc.github.io/ELMO/)
[Character embedding CNN](https://towardsdatascience.com/the-definitive-guide-to-bidaf-part-2-word-embedding-character-embedding-and-contextual-c151fc4f05bb)
[The Illustrated BERT EMLO and co.](http://jalammar.github.io/illustrated-bert/)
[Transfer learning in NLP](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5888218f39_177_4)
[VisualGuideToUsingBERT](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
[An In-Depth Tutorial to AllenNLP](https://mlexplained.com/2019/01/30/an-in-depth-tutorial-to-allennlp-from-basics-to-elmo-and-bert/)
[Transfer learning using elmo embedding](https://towardsdatascience.com/transfer-learning-using-elmo-embedding-c4a7e415103c)
[State of transfer learing in NLP](https://ruder.io/state-of-transfer-learning-in-nlp/)
[Generalized language model: ULMfit&openai GPT](https://www.topbots.com/generalized-language-models-ulmfit-openai-gpt/)
[Bert模型及fine-tuning](https://zhuanlan.zhihu.com/p/46833276)
[Openai GPT2 详解](https://zhuanlan.zhihu.com/p/57251615)
[How to make custom AI-generated text with GPT2](https://minimaxir.com/2019/09/howto-gpt2/)
[GPT2: Understand language generation through visualization](https://towardsdatascience.com/openai-gpt-2-understanding-language-generation-through-visualization-8252f683b2f8)
[GPT为什么不能双向？](https://www.zhihu.com/question/322034410/answer/794201004)
[📚The Current Best of Universal Word Embeddings and Sentence Embeddings](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)
[🦄 How to build a State-of-the-Art Conversational AI with Transfer Learning](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313)
[Practical Applications of Open AI’s GPT-2 Deep Learning Model](https://medium.com/the-research-nest/practical-applications-of-open-ais-gpt-2-deep-learning-model-14701f18a432)
[Unsupervised NER with BERT](https://www.quora.com/q/idpysofgzpanjxuh/Unsupervised-NER-using-BERT)
[BERT explained: State of the art language model for NLP](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
[Zero shot GPT2](https://rakeshchada.github.io/Zero-Shot-GPT-2.html)
[Practical Applications of Open AI’s GPT-2 Deep Learning Model](https://medium.com/the-research-nest/practical-applications-of-open-ais-gpt-2-deep-learning-model-14701f18a432)
[Understanding BERT Part 2: BERT Specifics](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73)
[Google BERT — Pre Training and Fine Tuning for NLP Tasks](https://medium.com/@ranko.mosic/googles-bert-nlp-5b2bb1236d78)
[why BERT has 3 embedding layers?](https://mc.ai/why-bert-has-3-embedding-layers-and-their-implementation-details/)
[from-pre-trained-word-embeddings-to-pre-trained-language-models-focus-on-bert](https://towardsdatascience.com/from-pre-trained-word-embeddings-to-pre-trained-language-models-focus-on-bert-343815627598)
[google BERT - pretraining and finetuing for NLP tasks](https://medium.com/@ranko.mosic/googles-bert-nlp-5b2bb1236d78)
[NLP: Explaining Neural language model](https://mchromiak.github.io/articles/2017/Nov/30/Explaining-Neural-Language-Modeling/#.XniDIWgzZPY)
[Bert微调技巧实验大全](https://zhuanlan.zhihu.com/p/109143667)
[BERT finetune的艺术](https://zhuanlan.zhihu.com/p/62642374)
[Bert在NLP各领域的应用进展](https://zhuanlan.zhihu.com/p/68446772)
[GPT2 finetune @familiarcycle.net/](https://familiarcycle.net/)
[paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained](https://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/)
<!--stackedit_data:
eyJoaXN0b3J5IjpbNTQyMDMyODA1LDgwMDczMjU3NCwtMTgyMz
Y5MTI3OCwtNjAwNDkxMjQzLC02MTA1Mzk3MTUsMzEzNjM3ODcx
LC05MDc5NDI3OTIsLTIwMDYzNzE4ODQsODc0MjQ3MTgzLC02OD
M5OTMxNjYsLTM3MDI5MjIzOSwxNzIzMTQzNjc1LDE0NjQ4MTc5
Miw0NDUzMDM4NTksNjU1OTg2NTcwLC0yMDE5NDg4MjI3LDExNj
gxNTc4NzcsLTQ5NDI4MTA5OCwzNTEyODQzMiwtNjE0MTk3NzIx
XX0=
-->