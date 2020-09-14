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
6. 总结


self-supervised learning is important area because it can greatly reduce the effort of training deep model, 

作为NLP迁移学习的成功应用，BERT证明了。。。本文旨在介绍BERT模型的结构和设计原理，以及BERT的应用。
## 迁移学习和预训练模型
![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vR6JBirfomJ2dxM1GDEl2GUZOXZeuyqcjRr7w6-t-s2vloOyAZk8GTRP1IyVmczcmyEINONHs5DhpH0/pub?w=593&h=343)

深度学习由于在处理复杂特征（图像，声音，文本）的任务上相比传统机器学习方法有巨大的优势，获得了越来越多的关注和发展。随着任务复杂度的增加，深度学习模型呈现出越来越复杂的趋势。深度学习对训练数据的依赖非常强，这是由于复杂模型需要大量的数据才有可能的理解数据的潜在（复杂）特征。一般来说，模型越复杂就需要越多的数据进行训练。这就导致了比较复杂的深度模型需要海量的数据来进行训练。由于训练数据通常需要人工标记因此海量训练数据的获取成本非常高，这使得训练或者改进深度模型成为耗时耗力的过程，非常不利于深度模型的推广和应用。

~~>**训练数据不足**是一些特殊领域中不可避免的问题。数据的收集是复杂和昂贵的，这使得构建大规模、高质量的带注释的数据集非常困难。例如，生物信息学数据集中的每个样本经常显示一个临床试验或一个痛苦的病人。此外，即使我们付出了昂贵的代价来获取训练数据集，也很容易过时，不能有效地应用于新的任务中。~~

为了解决这个问题，人们尝试将深度学习过程中产生的具有共性的知识提取出来用于类似目标的机器学习任务中去，这样。。就可以“站在巨人的肩膀上”而不必从零开始，从而节省了大量的资源和时间，这就是迁移学习(Transfer learning)的基本思想。基于这种重用的思想，迁移学习将一个完整的训练任务分成了两个阶段：预训练和微调训练

迁移学习是在源任务模型和新任务模型具有相关性的前提下，把已经训练好的模型参数迁移到新的模型来帮助新模型训练，这样就可以在源任务模型的基础上针对新任务进行调整和改进，而不必从零开始，从而节省大量的时间和金钱。

~~>迁移学习放松了训练数据必须与测试数据独立且同分布(i.i.d)的假设，激励我们利用迁移学习来解决训练数据不足的问题。在迁移学习中，训练数据和测试数据不需要是i.i.d。不需要对目标域内的模型进行从零开始的训练，可以显著降低对目标域内训练数据和训练时间的需求。~~
![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vStoAwye3EraSC6HH5m_S8VOsVEp3hsTtQuAVF-dEmPlFvEZqAxBHDQryl3FnVf_BZ6Csb969AGbChe/pub?w=791&h=385)
- 预训练阶段
这个阶段的训练目标是生成包含可重用知识的模型-预训练模型。预训练这是一个耗时耗力的巨大工程，为了使得更多不同的任务能从中受益，人们追求更加通用的预训练模型，由于通用知识的复杂性，预训练模型都非常复杂。而这类复杂模型只能靠海量来进行训练，这个阶段会耗费大量的计算资源。除了数据量要求大之外，预训练对数据的质量也有较高要求，例如在CV领域最成功的迁移学习的的应用是imagenet训练数据集及建立在其基础之上的预训练模型，如VGG19， ResNet50。预训练采用监督式训练，即每个imagenet数据集中的图片都有一个人工标注的描述该图片所属类型的标签。Imagenet将超过一千四百万图片通过众包的方式进行人工标注，将他们分成2万多个不同分类，这项从2007年开始的浩大工程为计算机视觉图形相关的预训练提供了高质量的训练数据，从而为CV迁移学习打下了基础。
- 微调阶段
根据任务的需要，在预训练模型的基础上设计并加入相应的模型结构，比如。。。。再使用任务相关的少量训练数据来调整模型参数使其适应该任务。

### NLP的迁移学习
NLP的迁移学习同样分为预训练和微调两步，预CV任务不同的是在预训练阶段NLP采用了自监督学习（self supervised learning）方式，这是由于NLP中的基本元素-word（或字）的含义通常由其所在的语句的上下文来决定，具有高度的灵活性，无法像CV中那个用一个固定的标签来标记。所幸的是使用语言模型可以很好地利用现有文本资料使用自监督学习的方式来进行预训练。

由于NLP主要关注语言（字符序列）的理解和处理，作为语言基本组成单位的词（word）也就自然成为了预训练的关注点。预训练的目标经历逐步的发展变化
#### 预训练 pre training
#####  训练目标: 生成词（字）编码 word embedding
- 静态词编码（static word embedding），这是一类早期的固定编码方式，比Word2Vec，Glove等，顾名思义这类编码赋予每个词固定的编码值，并且编码值体现了词的代表的含义，我们可以通过对编码值的运算得到有意义的结果，比如著名的例子 ***king — man + woman = queen***

- 语境词编码（contextualized word embedding），静态词编码的最大的问题在于它只能个每一个词一个编码值，无法处理一词多义的情况。将“我爱吃苹果”和“我爱苹果手机”中的苹果赋予相同的编码是不合适的，更合理的方式是通过结合词出现的上下文判断词的含义，比如通过“吃”和“手机”来判断上面两句话中的“苹果”分别代表一种水果和一个品牌，这就是语境词编码的基本思想。所以从使用者角度来说，我们需要一个模型能过通过输入语句得到（计算出）该语句的含义，或者该语句中每个词的含义。从这个意义上讲，我们本质上需要的是一种能够提取语义特征的能力，这和CV中的迁移学习的目标是一致的。
	- 单向语境编码 LSTM
	- 双向语境编码 elmo
		
#### 训练方式  self-supervised learning

由于语言的动态特性，NLP任务
> NLP的最大挑战之一是缺乏足够的培训数据。总体而言，有大量文本数据可用，但是如果我们要创建特定于任务的数据集，则需要将该堆划分为很多不同的字段。而当我们这样做时，我们最终仅得到数千或数十万个人标记的培训示例。不幸的是，为了表现良好，基于深度学习的NLP模型需要大量的数据-在数百万或数十亿的带注释的训练示例上进行训练时，他们看到了重大改进。为了帮助弥合数据鸿沟，研究人员开发了各种技术，可在网络上使用大量未注释的文本来训练通用语言表示模型（这称为预训练）。然后，可以在较小的特定于任务的数据集上微调这些通用的预训练模型，例如，在处理诸如问题回答和情感分析之类的问题时。与从头开始对较小的特定于任务的数据集进行训练相比，此方法可显着提高准确性。

同样的为了NLP领域也由类似的需求：为每个词建立正确的标签数据来帮助进行监督训练，根据语言的特点，设计了语言模型（Language Model）这种训练任务来进行。。。LM属于自监督（self supervised）训练方法，使用这种训练方法不需要为语句进行人工标注，而只使用语句序列本身就可以进行训练。LM是一种统计方法，用于计算一个序列$W$（由词$w_i, w_2, ... w_m$组成的一句话）出现的概率$$P(W)=P(w_1,w_2,w_3,...w_m)$$LM也可以用于计算在一个序列中某个词$w_{n+1}$出现的概率$$P(w_{n+1}|w_1,w_2, w_3,...w_n)$$
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

### 微调 fine tune
由于使用海量的数据进行预训练，预训练模型通常具有一般的常识，由此作为基础再进行微调，使得模型能更好的适合特定任务。
- 模型调整
通常做法是在预训练模型基础上增加任务相关的层，由于NLP的预训练模型通常是包含序列上下文的embeddings，如由全连接层和softmax运算构成的分类层用于分类任务。
![enter image description here](https://miro.medium.com/max/2248/1*GVcm-gUJ5r6niWB6OsOg_w.png)
- supervised learning
使用少量任务相关的标记数据来进行微调，通常的做法是在预训练模型的后面直接加上上一个分类器（由全连接和softmax运算构成）使模型输出一个预测类型，计算cross entropy误差从而通过反向传递更新模型参数。
	- ~~更新全部模型参数~~
	- 只更新任务层参数 - 预训练模型只作为特征提取器

## BERT简介
BERT（Bidirectional Encoder Representations from Transformer），同他名字说的一样，BERT是一个利用Transformer实现的双向编码器，  用于提取输入序列特征信息的预训练模型。When BERT was published it achieved [state-of-the-art] performance in 11 [natural language understanding] tasks:[[1]] [GLUE]task set (consisting of 8 tasks), [MultiNLI] [SQuAD] v1.1, SQuAD v2.0
2018, google发表了论文BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding， 2019年google将BERT模型应用到了搜索服务中，现在已经支持了超过70种语言


**bidirectional <-> LM 的矛盾如何解决？ MLM+NSP** 

BERT模型生成的元素编码属于动态的双向语境编码，它能根据输入序列生成每个序列元素（word）在序列上下文中的特征向量， 与ELMO不同的是，它基于注意力机制（attention mechanism）, 利用Transformer强大的特征提取能力，实现了深度双向语境编码，这也是BERT的区别于传统的双向编码技术（如ELMO）最大创新之处。这里的**深度双向**是指超过传统的双向编码技术，由于BERT使用的Transformer编码器以Attention机制为基础，注意力机制可以一次看到所有的序列元素，每个元素的编码的计算都包含了该元素之前和之后的序列信息，从方向来说，同时包含了之前和之后两个方向，从距离来讲，同时计算所有元素也就代表包含了不同距离。从编码方式上，attention机制可以直接  elmo独立的进行前项和后向的编码，只是简单进行前向编码的相加。

    这与简单的双向编码的只能一次一步的


![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vQlNArvjK1O_E3Ynd7lKUf0ux-PylQpDm4nH8swVZ76_qxu4Ch-jTz0zIrAN5-ytq937ZGabG4VOTAp/pub?w=968&h=343)

虽然attention就有全局视野，但也不是完全不能使用LM方式进行训练，比如GPT使用了遮罩的方式使模型无法看到当前元素之后的序列信息，因此它属于单向语言模型。


### BERT预训练模型结构

BERT模型主要包含这个部分，编码层和Transformer编码器
![enter image description here](https://www.lyrn.ai/wp-content/uploads/2018/11/transformer.png)

#### 编码层
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
- why 512?
>Theoretically there is nothing restricting a Transformer to have greater sequence length. Practically, there are resource constraints - especially memory complexity when doing self-attention which is quadratic in terms of sequence length. Another reason why BERT is restricted to 512 may be because that was the sequence length it was originally restricted to while training but I am not sure.
>[https://github.com/google-research/bert/issues/27](https://github.com/google-research/bert/issues/27)
>[https://github.com/google-research/bert/issues/66](https://github.com/google-research/bert/issues/66)
>We don't plan to make major changes to this library, so anything like that would be part of a separate project.
Our recommended recipe is exactly what you describe (it's what we do for SQuAD), but you can actually fine-tune on it normally (we just don't do it for SQuAD because only a few percent of SQuAD documents are longer than 384 do so it didnt matter. But we should have).
Let's say you have:
`the man went to the store and bought a gallon of milk`
And had  `max_seq_length = 6, stride = 3`, then you could split it up like this:
```
the man went to the store
to the store and bought a
and bought a gallon of milk
```
>So from  `BertModel`'s perspective this is a 3x6 minibatch, but crucially you can reshape it after you get it back from  `BertModel.get_sequence_output()`  and softmax over all the tokens when you compute the loss (with some masking to make sure you don't double count the boundary words like  `to the store`  and  `and bought a`). So you will be fine-tuning over the whole document end-to-end. The exact implementation is task-specific of course.

#### Transformer编码器
Transformer模型是由google ai于2017年发布的一个编码器-解码器架构模型，最初应用于机器翻译。Transformer的最大特点是使用注意力机制（attention mechanism），解决了使用RNN模型造成的梯度爆炸和无法并行的问题，并且实践证明transformer中提出的多头注意力具有强大的特征提取能力，性能超越了RNN,CNN等传统方法。
> Transformer所使用的注意力机制的核心思想是去计算一句话中的每个词对于这句话中所有词的相互关系，然后认为这些词与词之间的相互关系在一定程度上反应了这句话中不同词之间的关联性以及重要程度。因此再利用这些相互关系来调整每个词的重要性（权重）就可以获得每个词新的表达。这个新的表征不但蕴含了该词本身，还蕴含了其他词与这个词的关系，因此和单纯的词向量相比是一个更加全局的表达。
> Transformer通过对输入的文本不断进行这样的注意力机制层和普通的非线性层交叠来得到最终的文本表达。

Transformer由编码器和解码器组成，编码器负责将输入序列中的每个元素（word）转换为包含上下文信息的特征向量，再由解码器根据编码后的特征向量生成输出序列。BERT模型中只使用了transformer的编码器，它主要由若干个结构相同的编码层连接而成。每一个编码层主要有一个多头自注意力计算单元（Multi-Head Attention）和按位前馈网络(Feed Forward)组成，多头自注意力计算单元负责为每个输入元素生成特征向量，前馈网络能够通过组合元素特征向量生成更复杂的特征向量。

![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vSqp25HORnsDrfUfkTFUgKeTC7IITVZrTMXBuf6eSp4_HmCsGRoGwAxEoN87fuhT98Xsc4IulE_U4vM/pub?w=960&h=720)

### BERT预训练模型的能力
无需任何训练也可完成简单的问答
## BERT的预训练
训练数据: 
- BooksCorpus (800M words)
- EnglishWikipedia (2.5B words)

### 任务设计
自然语言中蕴含的意义是以句子的形式表现的，为了让预训练模型学习到更多的知识，BERT的预训练被设计为多任务学习（multi-task learning），包含两个任务：一个是 Masked Language Model(MLM)，训练模型理解单个句子蕴含的意义；另一个是 Next Sentence Prediction(NSP)，训练模型理解不同语句之间的上下文含义。通过这两个任务的联合学习，可以使得 BERT 学习到的表征既有 token 级别信息，同时也包含了句子级别的语义信息。
#### Masked Language Model  - MLM
之所以BERT使用重新设计的MLM，是由于注意力机制的使用使得BERT模型能够同时“看到”所有的序列元素，因此无法使用传统语言模型通过预测下一个元素的方式来进行训练。因此BERT使用了Mask LM，做法是随机挑选序列中的若干元素，将他们遮（mask）起来，使注意力机制无法“看到”，通过训练模型预测未知元素的值来促使模型学习到整个序列的含义。Mask LM训练的思路类似于填词游戏，即通过上下文的信息来判断模型被隐藏的词。从原理上讲这种方式可以很好的匹配注意力机制的运算方式，但是在实现训练中遮罩元素的数量对于训练的效果和速度都有很大影响，如果mask太多，会丢失context，如果mask太少，训练太慢。
>[https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)

BERT的做法是给定一个句子，随机遮住15%的词，然后输入BERT模型并让BERT来预测这些被遮住的词。具体来说，对于每个被遮住的词，执行下面的遮罩替换规则：
-   以80%的概率替换成[MASK]，比如my dog is hairy → my dog is [MASK]
-   以10%的概率替换成随机的一个词，比如my dog is hairy → my dog is apple
-   以10%的概率替换成它本身，比如my dog is hairy → my dog is hairy

>BERT is designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context.
BERT is [MASK1] to help **milk** understand the meaning of ambiguous language in text by using **surrounding** text to [MASK2] context
任务目标： 预测所有[MASK] 以及milk和surrounding位置上的词
测试数据：[MASK1]=designed, milk=computers, surrounding=surrounding, [MASK2]=establish

对于被遮罩词的替换规则
 - 如果只做[MASK]替换，预训练模型会被训练为对[MASK]进行预测，所以只会加强[MASK]附近上下文的分析而不是全部序列的分析。 而微调阶段的目标是分析整个序列，它的输入不包含[MASK]，与预训练模型的目标不一致，因此会导致预训练模型在微调阶段性能下降。
 - 为了更加符合微调阶段的目标，作者加入了一种新的预处理方式，即以10%的几率随机将原词computer替换为其他词milk而不是[MASK]，为了得出正确结果（computer）模型需要分析milk的上下文。由于所有的词都可能被替换，这就要求模型要对所有输入元素的上下文进行分析，从而满足微调的需要。
 - 考虑到如果只用[Mask]和任意词进行替换，模型会认为看到当前的词都是不真实的（替换过的），这会导致生成embedding的过程完全不参考当前词。为此预训练时也会也10%的概率使用原词替换（如surrounding），这样模型也会参考当前词来生成embedding。
 - 对于为何也80%，10%和10%的比例分别进行Mask，随机词和原词替换，作者的解释是基于经验设计的比例，可能存在效果更好的比例分布，但是最终结果应该相差不大。
 > We didn't try a lot of ablation on this. Those numbers are just what made sense to me and the only thing that I tried. It's possible that other values will work better (or more likely, the system isn't very sensitive to the exact hyperparameters).   [https://github.com/google-research/bert/issues/85](https://github.com/google-research/bert/issues/85)
- 最后，由于MLM只预测15%的序列元素，因此比标准LM训练速度要慢。

>_Why did they not use a ‘<MASK>’ replacement token all around?_
If the model had been trained on only predicting ‘<MASK>’ tokens and then never saw this token during fine-tuning, it would have thought that there was no need to predict anything and this would have hampered performance. Furthermore, the model would have only learned a contextual representation of the ‘<MASK>’ token and this would have made it learn slowly (since only 15% of the input tokens are masked). By sometimes asking it to predict a word in a position that did not have a ‘<MASK>’ token, the model needed to learn a contextual representation of  _all_  the words in the input sentence, just in case it was asked to predict them afterwards.
_Are not random tokens enough? Why did they leave some sentences intact?_
Well, ideally we want the model’s representation of the masked token to be better than random. By sometimes keeping the sentence intact (while still asking the model to predict the chosen token) the authors biased the model to learn a meaningful representation of the masked tokens.
_Will random tokens confuse the model?_
The model will indeed try to use the embedding of the random token to help in its prediction and it will learn that it was actually not useful once it sees the target (correct token). However, the random replacement happened in 1.5% of the tokens (10%*15%) and the authors claim that it did not affect the model’s performance.
_The model will only predict 15% of the tokens but language models predict 100% of tokens, does this mean that the model needs more iterations to achieve the same loss?_
Yes, the model does converge more slowly but the increased steps in converging are justified by an considerable improvement in downstream performance.
#### Next Sentence Prediction（NSP）
输入句子A和句子B，判断句子B是否是句子A的下一句，通过迭代训练，可以学习到句子间的关系，这对于文本匹配类任务显得尤为重要。
Next Sentence Prediction（NSP）的任务是判断句子B是否是句子A的下文。如果是的话输出’IsNext‘，否则输出’NotNext‘。训练数据的生成方式是从平行语料中随机抽取的连续两句话，其中50%保留抽取的两句话，它们符合IsNext关系，另外50%的第二句话是随机从预料中提取的，它们的关系是NotNext的。这个关系保存在图4中的`[CLS]`符号中。
>_Why is a second task necessary at all?_
The authors pre-trained their model in  _Next Sentence Prediction_  because they thought important that the model knew how to relate two different sentences to perform downstream tasks like question answering or natural language inference and the “masked language model” did not capture this knowledge. They prove that pre-training with this second task notably increases performance in both question answering and natural language inference.
### 预训练模型架构
为了完成为预训练的两个任务，模型需要进行相应的设计。总体来看，应用于预训练的模型可以分为两部分：
#### 预训练模型-BERT
这部分就是。。。预训练完成后这部分模型会包含可用于微调任务的通用知识，主要包含以下三部分：
- BertEmbeddings
	- segment embedding
	- position embedding
	- 
- BertEncoder
- BertPooler

#### 预训练层BertPretrainingHeads(MLM head+NSP head)
顾名思义，这层的目的是为了完成预训练任务，因此包含了两个模型结构分别用于进行MLM和NSP训练任务
	- MLM head
	![enter image description here](https://pic4.zhimg.com/80/v2-4364096101aad977b125aa585d187387_720w.jpg)
当把词汇表（vocabulary）中的每一个词都作为一个单独的类型时，对未知token进行预测就变成了在所有类型中判断可能性最高的分类，也这就是典型的多类型分类问题。为了对token进行分类判断，需要在BERT的输出上增加一个多类型分类器（在实现中被称为MLM head），它包含一个全连接网络和softmax运算，可以将通过BERT 编码器编码过的token转换为vocabulary长度个输出，每个输出代表属于对应分类的概率。应用这个分类器对所有token计算出每个类型的概率，再和这个token的真实分类进行比较，通过cross entropy函数计算误差。~~之所以对全部token进行分类预测的原因是由于被遮罩的词有15%的机率被替换成随机词，因此每个词都可能是被遮罩过的。~~
$$ L_{mlm}(\theta, \theta_m) = -\sum_{i=1}^M\log p(m=m_i| \theta, \theta_m)$$

	```
	masked_lm_loss = CrossEntropyLoss(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
	```
	- NSP head
![enter image description here](https://picb.zhimg.com/80/v2-33d191eee24be9a47b7799b939564d74_720w.jpg)
NSP的训练目标是判断两个句子是否是连续的，因此它也属于二元（是和否）分类问题。和MLM相似，需要加上一个二值分类器（NSP head）来进行类型判断。与MLM不同的是，由于【CLS】token包含了整个序列（包含两个句子）的含义，因此只需要对【CLS】token进行类型判断。预测误差和MLM一样使用cross entropy函数计算
$$L_{nsp}(\theta, \theta_n) = - \sum_{i=1}^N \log p(n=n_i|\theta, \theta_n)$$

	```
	next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
	```

>MLM这种设计的原因是由于BERT使用的注意力机制有全局的视野，能够一次同时访问序列的所有元素，因此无法使用传统的语言模型那种一步一看的训练方式。**前者用于建模更广泛的上下文，通过 mask 来强制模型给每个词记住更多的上下文信息；后者用来建模多个句子之间的关系，**
>![enter image description here](https://www.researchgate.net/profile/Jan_Christian_Blaise_Cruz/publication/334160936/figure/fig1/AS:776030256111617@1562031439583/Overall-BERT-pretraining-and-finetuning-framework-Note-that-the-same-architecture-in.ppm)

![enter image description here](https://miro.medium.com/max/412/1*ZLxPJwuHALDLTdqZfxhVNw.png)


### 损失函数
$$L(\theta, \theta_m, \theta_n) = L_{mlm} + L_{nsp}$$
```
total_loss = masked_lm_loss + next_sentence_loss
```
与任务相对应，BERT的损失函数由两部分组成，第一部分是来自 Mask-LM 的**单词级别分类任务**，另一部分是**句子级别的分类任务**。通过这两个任务的联合学习，可以使得 BERT 学习到的表征既有 token 级别信息，同时也包含了句子级别的语义信息。具体损失函数如下：

![[公式]](https://www.zhihu.com/equation?tex=L%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B1%7D%2C+%5Ctheta_%7B2%7D%5Cright%29%3DL_%7B1%7D%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B1%7D%5Cright%29%2BL_%7B2%7D%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B2%7D%5Cright%29)

其中  ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta)  ​ 是 BERT 中 Encoder 部分的参数，​  ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_1)  是 Mask-LM 任务中在 Encoder 上所接的输出层中的参数，​  ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_2)  则是句子预测任务中在 Encoder 接上的分类器参数。因此，在第一部分的损失函数中，如果被 mask 的词集合为 M，因为它是一个词典大小 |V| 上的多分类问题，那么具体说来有：

![[公式]](https://www.zhihu.com/equation?tex=L_%7B1%7D%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B1%7D%5Cright%29%3D-%5Csum_%7Bi%3D1%7D%5E%7BM%7D+%5Clog+p%5Cleft%28m%3Dm_%7Bi%7D+%7C+%5Ctheta%2C+%5Ctheta_%7B1%7D%5Cright%29%2C+m_%7Bi%7D+%5Cin%5B1%2C2%2C+%5Cldots%2C%7CV%7C%5D)

在句子预测任务中，也是一个分类问题的损失函数：

![[公式]](https://www.zhihu.com/equation?tex=L_%7B2%7D%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B2%7D%5Cright%29%3D-%5Csum_%7Bj%3D1%7D%5E%7BN%7D+%5Clog+p%5Cleft%28n%3Dn_%7Bi%7D+%7C+%5Ctheta%2C+%5Ctheta_%7B2%7D%5Cright%29%2C+n_%7Bi%7D+%5Cin%5B%5Ctext+%7BIsNext%7D%2C+%5Ctext+%7BNotNext%7D%5D)

因此，两个任务联合学习的损失函数是：

![[公式]](https://www.zhihu.com/equation?tex=L%5Cleft%28%5Ctheta%2C+%5Ctheta_%7B1%7D%2C+%5Ctheta_%7B2%7D%5Cright%29%3D-%5Csum_%7Bi%3D1%7D%5E%7BM%7D+%5Clog+p%5Cleft%28m%3Dm_%7Bi%7D+%7C+%5Ctheta%2C+%5Ctheta_%7B1%7D%5Cright%29-%5Csum_%7Bj%3D1%7D%5E%7BN%7D+%5Clog+p%5Cleft%28n%3Dn_%7Bi%7D+%7C+%5Ctheta%2C+%5Ctheta_%7B2%7D%5Cright%29)

>### 预训练技巧
>具体的预训练工程实现细节方面，BERT 还利用了一系列策略，使得模型更易于训练，除了常用的layer normalization，dropout之外，还有对于学习率的 warm-up 策略，使用的激活函数不再是普通的 ReLu，而是 GeLu。
>- Transformer related :  dropout, layer_norm, residual
 
### 预训练流程
预训练的目的是生成能够给下游任务使用的通用模型，因此BERT在预训练中加入两个特殊token，CLS和SEP。
CLS加在输入序列的开头，它也参与Transformer计算。我们知道注意力计算是对所有元素以一定的权重进行加权平均，由于CLS本身不包含任何意义，因此与序列中的其他元素都不相关，因此CLS token通过注意力运算的结果是将所有元素的意思以相似的权重进行加权平局，这也就是整个序列的unbias意义。由于CLS embedding包含了这个序列的含义，因此在对序列进行分类等微调任务中会直接对CLS embedding进行分类训练。
另一个特殊token是SEP，当输入序列中包含多个句子时，使用这个token分隔不同的句子。和CLS不同的是，SEP embedding本身不会用于微调任务，它主要用于预训练中的NSP子任务。
>The pre-training corpus was built from BookCorpus (800M words) and English Wikipedia (2,500M words). Tokens were tokenized using 37,000 WordPiece tokens.
To generate the pre-training sequences, the authors got random samples in batches of two (50% of the time adjacent to each other) such that the combined length of the two chosen sentences was ≤512 tokens. Once each sequence was built, 15% of its tokens were masked.
An example of a pre-training sequence presented in the paper is:
> > Input = [CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]
In this case the sentences are adjacent, so the label in [CLS] would be ‘<IsNext>’ as in:
> > Input = <IsNext> the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]
The loss was calculated as the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood.

![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vRFdq5CGCgn5WdAHdz88Z5ePsIU58vHz0HVYx56PQ3TP7Xi2WAbSkAbWx1Q4VA8ZkJ3mpSvlpmV1v-0/pub?w=1746&h=911)
[http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)  Recapping a sentence’s journey
Each training data contains Two sentences, $W_1[w_{11}, w_{12}, w_{13}, w_{14}, w_{15}], W2[w_{21}, w_{22},w_{23},w_{24},w_{25}]$
1. 预处理: 
    1.0  tokenization (WordPiece)
	1.1 . 加入特殊符号CLS和SEP： [CLS] BERT is awesome. [SEP] I love BERT. [SEP]"
	1.2.  Add masks to some words: [CLS] BERT [MASK] awesome. [SEP] I love BERT. [SEP
	1.3.  Generate pretrain data
		1.3.1 for NSP:  50% isNext, 50% isNotNext, true_label_{NSP}=
		1.3.2 for MLM: {tokens:["CLS", "BERT", "MASK", "awesome", "SEP"], masked_token:{index:2, value:"is"}}
		
2. Embedding
	2.1 word embedding(WE):  wordpiece tokenization (shape=(vocab_size * hidden_size))
	2.2 positional embedding(PE): (shape=(max_position * hidden_size))
	2.3 segment embedding(SE): (shape=(segment_size * hidden_size))
	E = WE + PE + SE
3. Transformer编码: 
    Bert embeddings = Transformer(E)
4. 预测
	MLM_prediction = multi_class_predictor(BERT_embeddings)
	multi_class_predictor = softmax(fully_connected(BERT_embedding_length, size(vocabulary))
	
	NSP_prediction = binary_class_predictor([CLS]_embeddings)
	binary_class_predictor = softmax(fully_connected(BERT_embedding_length,2))
	
5. 计算误差
	MLM_loss = cross_entropy(MLM_prediction, true_mlm_labels)
	NSP_loss = cross_entropy(NSP_prediction, true_nsp_label)
	BERT_loss = MLM_loss + NSP_loss
	

## BERT的微调（fine tuning）
综上所述，BERT的重要意义不仅是架构上的改进，通过海量数据训练出来的BERT预训练模型更具实践意义，它使得不具备大量数据和计算资源的普通人也能够站在巨人的肩膀上，将其蕴含的强大通用知识运用在自己的机器学习任务中。
微调是在BERT预训练模型的基础上加入针对性的微调网络结构（微调层），并利用（相对）少量数据来训练微调层使之能够满足一定的任务需要的学习过程。在微调训练中，BERT预训练模型的参数一般不会被训练（更新），BERT预训练模型网络只起到特征提取的作用。
从本质上讲，基于transformer的BERT模型能够为序列元素生成具备上下文信息的编码，围绕着这个特点，作者设计了很多微调任务（如NER，STS，等），相对于BERT的预训练中使用的MLM和NSP任务，可以将这些微调任务分为两类：单句上下文任务和多句上下文任务。下面我们介绍这两类任务的典型代表。
> 利用BERT预训练模型进行微调的基本思路是利用BERT模型已经学好的参数进行。。。
BERT模型在设计时就考虑到了方便下游任务，它可以应用在很多类新的
**方法：固定预训练模型的参数，训练微调层的参数**

![enter image description here](https://www.topbots.com/wp-content/uploads/2019/12/cover_transfer_learning_1600px_web.jpg)
![enter image description here](https://www.researchgate.net/profile/Jan_Christian_Blaise_Cruz/publication/334160936/figure/fig1/AS:776030256111617@1562031439583/Overall-BERT-pretraining-and-finetuning-framework-Note-that-the-same-architecture-in.ppm)

![enter image description here](https://lilianweng.github.io/lil-log/assets/images/BERT-downstream-tasks.png)
### 单句分类问题

![](https://docs.google.com/drawings/d/e/2PACX-1vStZ1DhjhFi-1LIWS7hCHsvTzu_SQ-DS9Dh6MDCC2qcjmmDcuWte2Ii3wATxx149_NX-OOVzU1nqbr9/pub?w=884&h=709)
- 任务介绍
这种类型的任务对输入的一句话进行类型判断，例如分析一条购买评价的语义是肯定的还是否定的。

- Make use of the CLS token
- 微调层结构：分类器（全连接+softmax）[https://github.com/huggingface/transformers/blob/c67d1a0259cbb3aef31952b4f37d4fee0e36f134/src/transformers/modeling_bert.py#L1234-L1241](https://github.com/huggingface/transformers/blob/c67d1a0259cbb3aef31952b4f37d4fee0e36f134/src/transformers/modeling_bert.py#L1234-L1241)

    class BertForSequenceClassification(BertPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

[https://github.com/huggingface/transformers/blob/c67d1a0259cbb3aef31952b4f37d4fee0e36f134/src/transformers/modeling_bert.py#L1291-L1299](https://github.com/huggingface/transformers/blob/c67d1a0259cbb3aef31952b4f37d4fee0e36f134/src/transformers/modeling_bert.py#L1291-L1299)
- 训练数据
	- $x={x_1, x_2, x_3, ... , x_n}, y=label$
- 训练流程
- 损失函数：cross-entropy
- 
预测

    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    loss, logits = outputs[:2]

>### 语义相似度分析
>输入两句话，分析他们的语义是相似的还是不同的。
预处理
![](https://pic1.zhimg.com/80/v2-971f887ed616ea0f65941c8dc15ee128_720w.jpg)
  实际操作时，上述最后一句话之后还会加一个[SEP] token，语义相似度任务将两个句子按照上述方式输入即可，之后与论文中的分类任务一样，将[CLS] token位置对应的输出，接上softmax做分类即可(实际上GLUE任务中就有很多语义相似度的数据集)。
  微调层：

>### 多标签分类 NER
>多标签分类任务，即MultiLabel，指的是一个样本可能同时属于多个类，即有多个标签。以商品为例，一件L尺寸的棉服，则该样本就有至少两个标签——型号：L，类型：冬装。
对于多标签分类任务，显而易见的朴素做法就是不管样本属于几个类，就给它训练几个分类模型即可，然后再一一判断在该类别中，其属于那个子类别，但是这样做未免太暴力了，而多标签分类任务，其实是可以**只用一个模型**来解决的。
利用BERT模型解决多标签分类问题时，其输入与普通单标签分类问题一致，得到其embedding表示之后(也就是BERT输出层的embedding)，有几个label就连接到几个全连接层(也可以称为projection layer)，然后再分别接上softmax分类层，这样的话会得到​  ![[公式]](https://www.zhihu.com/equation?tex=loss_1%2C%5C+loss_2%2C%5C+%5Ccdots%2C%5C+loss_n)  ，最后再将所有的loss相加起来即可。这种做法就相当于将n个分类模型的特征提取层参数共享，得到一个共享的表示(其维度可以视任务而定，由于是多标签分类任务，因此其维度可以适当增大一些)，最后再做多标签分类任务。~~

### 问答问题 SQuAD
![](https://docs.google.com/drawings/d/e/2PACX-1vSGC-bIM19eOTk96audRUt_wo5hNgsIQpPZVo0pRfMLLxmCpYihvwaKYtcHI2Eyikmk4amBFMSNur6T/pub?w=835&h=686)
- 任务介绍
预测answer span(start pos, end pos)
- 模型架构
	- BERT预训练模型
	- SQuaD任务层
- 训练数据
	- question
	- reference
	- answer_start_pos
	- answer_end_pos
- 训练流程
	- question + [SEP] + reference
	- 
Use classification head for each token
can deal with looooong senquence？（>512）: 
[https://github.com/google-research/bert/issues/66](https://github.com/google-research/bert/issues/66)
how to get the context vector?

- **文本生成？NO!**
remember BERT does not include decoder?
- Bert use transformer as encoder, there is no decoder in BERT
- 

## BERT的改进
[关于BERT的若干问题整理记录](https://zhuanlan.zhihu.com/p/95594311)
### task design
- spanBERT [https://zhuanlan.zhihu.com/p/75893972](https://zhuanlan.zhihu.com/p/75893972)
### distillation

## 总结

BERT的核心思想是使用Transformer来进行深度双向上下文的语义分析，但是Transformer是一把双刃剑，它一方面提供了强大深度双向处理能力，而一方面也使传统的语言模型LM训练方法收到了影响。  由于深度双向会导致。。而无法使用LM进行训练，作者利用了MLM并设计了相应的预处理来解决预训练和微调训练的冲突。。。
BERT最大的成功在于提供了一个方便可用的预训练模型

从模型的角度来讲，为了让预训练模型服务于更多的下游任务，加入pooler

从任务设计的角度，作者MLM， NSP






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



~~### LAMP？not a BERT improvement~~
## BERT应用
### imageBert
### codeBert
## BERT in action
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
[Understanding BERT part2](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73)
[BERT源码分析](https://blog.csdn.net/weixin_37947156/article/details/94885499)
[BERT author explain BERT](https://www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_of_deep_bidirectional/)
[Examining BERT's raw embeddings](https://towardsdatascience.com/examining-berts-raw-embeddings-fd905cb22df7)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0NDAwODg0NDcsMTI4Nzk5NTk5NiwtNz
kxMTAxMjYyLC0zNTY0NDkwMDUsLTY2MTQ3ODk1NCwtMTczODcz
MjMyMCwtMjI4MTQxOTc2LC0xNDc5NTc3MDY4LC0xMTAyNTI3Nz
YzLDE4OTcxNjQxNDgsLTE2NDg0MjEzNDMsLTE4OTA4OTIwNiwx
ODg4MDMyNTAzLDIwMTY1NDY0NDIsMTY1MTk5ODk1NywtMjc1OT
I5NTgxLDg1NDg5MjEzOSwtMTMyNTcyMjc3MiwtMTY5MzU3MjYx
MSwtOTcxOTQ0ODBdfQ==
-->