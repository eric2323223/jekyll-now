# NLP transfer learning 

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
# Training objective of self-supervised learning - From Word2Vec to Elmo to Bert to XLNet

self-supervised learning is important area because it can greatly reduce the effort of training deep model, 

作为NLP迁移学习的成功应用，BERT证明了。。。本文旨在介绍BERT模型的结构和设计原理，以及BERT的应用。
## 迁移学习和预训练模型
![enter image description here](https://miro.medium.com/max/3283/1*Z11P-CjNYWBofEbmGQrptA.png)
迁移学习旨在通过重用 。。。来加速学习和增强预测的准确性，对于当今越来越复杂的神经网络来说，需要巨大的人力物力和时间成本。。。使用迁移学习是非常有意义的。通过再imagenet训练视觉特征提取网络，数据比较从头训练和使用迁移训练。。。
现实的问题是获取足够的标记数据非常困难，因此
### NLP的迁移学习
我们知道在CV中的迁移学习过程是首先训练一个通用的的图像特征提取模型（如VGG19， ResNet50等），再结合下游任务需要通过扩展第一阶段的模型来进行fine tuning。进行与CV任务类似，应用迁移学习解决NLP问题也可以分为两个阶段。首先通过预训练学习出可重用的特征提取模型，也叫预训练模型。由于NLP主要关注语言（字符序列）的理解和处理，作为语言基本组成单位的词（word）也就自然成为了预训练的关注点。预训练的目标经历逐步的发展变化
#### 预训练
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
	
	
#### 微调 fine tune
由于使用海量的数据进行预训练，预训练模型通常具有一般的常识，由此作为基础再进行微调，使得模型能更好的适合特定任务。微调工作可以以下两种形式：
- 监督式微调supervised fine tuning
使用少量任务相关的标记数据来进行微调，通常的做法是在预训练模型的后面直接加上上一个分类器（由全连接和softmax运算构成）使模型输出一个预测类型，计算cross entropy误差从而通过反向传递更新模型参数。
- **无监督式微调unsupervised fine tuning**?
- zero shot learning
无微调适用于容量更大预训练模型，这类模型一般包含了更多的常识，比如GPT2使用了xx的高质量数据进行预训练，无需微调也可能在不同下游任务重生成可接受的预测。对于这类模型，只需要给出少量的样例让模型理解预测意图。。。

## BERT简介
When BERT was published it achieved [state-of-the-art] performance in 11 [natural language understanding] tasks:[[1]] [GLUE]task set (consisting of 8 tasks), [MultiNLI] [SQuAD] v1.1, SQuAD v2.0
2018, google发表了论文BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding， 2019年google将BERT模型应用到了搜索服务中，现在已经支持了超过70种语言
BERT（Bidirectional Encoder Representations from Transformer）是一个预训练模型，它可以提取输入序列的上下文信息，

BERT最大的创新是将Transformer模型应用到了语言模型中，。。。。影响和决定了BERT很多特殊性质。在BERT之前，

- context dependent embedding
BERT模型生成的元素编码属于动态编码，它能根据输入序列生成每个序列元素（word）在序列上下文中的特征向量
- bidirectional Language Model
这是由于它是以Attention机制为基础。注意力机制可以一次看到所有的序列元素，每个元素的编码的计算都包含了该元素之前和之后的序列信息，因此BERT属于双向语言模型，并且由于能够同时看到前向和后向的信息，BERT不同于以往的双向语言模型，如ELMO，。。。。。。
并非所有的基于attention机制的模型都是双向语言模型，比如GPT使用了遮罩的方式使模型无法看到当前元素之后的序列信息，因此它属于单向语言模型。

- 

## BERT模型结构
### Transformer encoder based
BERT模型主要包含这个部分，编码层和Transformer编码器
![enter image description here](https://miro.medium.com/max/1095/0*ViwaI3Vvbnd-CJSQ.png)

### 编码层
编码层的作用是
1. 将输入语句（BERT is powerful）转换为模型可处理的浮点数向量
2. 加入特殊符号[CLS][SEP]

    embeddings = inputs_embeds + position_embeddings + token_type_embeddings

[https://mc.ai/why-bert-has-3-embedding-layers-and-their-implementation-details/](https://mc.ai/why-bert-has-3-embedding-layers-and-their-implementation-details/)
![enter image description here](https://i.stack.imgur.com/QCcYF.png)
- 词编码(config.vocab_size, config.hidden_size, padding_idx=0)
[https://www.topbots.com/generalized-language-models-bert-openai-gpt2/#input-embedding](https://www.topbots.com/generalized-language-models-bert-openai-gpt2/#input-embedding)
- 段编码(config.type_vocab_size, config.hidden_size)
由于BERT可以处理1或2条语句，用于区分不同语句
- 位置编码(config.max_position_embeddings, config.hidden_size)
不同于Transformer的基于周期函数的固定位置编码方法，BERT采用可学习的位置编码方式，bert中的最大句子长度是512 所以Position Embedding layer 是一个size为（512，768）的lookup table
### Transformer编码器
Transformer模型是由google ai于2017年发布的一个编码器-解码器架构模型，最初应用于机器翻译。Transformer的最大特点是使用注意力机制（attention mechanism），解决了使用RNN模型造成的梯度爆炸和无法并行的问题，并且实践证明transformer中提出的多头注意力具有强大的特征提取能力，性能超越了RNN,CNN等传统方法。
Transformer由编码器和解码器组成，编码器负责将输入序列中的每个元素（word）转换为包含上下文信息的特征向量，再由解码器根据编码后的特征向量生成输出序列。BERT模型中只使用了transformer的编码器，它主要由若干个结构相同的编码层连接而成。每一个编码层主要有一个多头自注意力计算单元和按位前馈网络组成，多头自注意力计算单元负责为每个输入元素生成特征向量，前馈网络能够通过组合元素特征向量生成更复杂的特征向量。
### 输出

## BERT的预训练
### 任务设计
BERT的预训练被设计为多任务学习（multi-task learning），包含两个任务：
- MLM
[https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
Training the language model in BERT is done by predicting 15% of the tokens in the input, that were randomly picked. These tokens are pre-processed as follows — 80% are replaced with a “[MASK]” token, 10% with a random word, and 10% use the original word. The intuition that led the authors to pick this approach is as follows (Thanks to Jacob Devlin from Google for the insight):

-   If we used [MASK] 100% of the time the model wouldn’t necessarily produce good token representations for non-masked words. The non-masked tokens were still used for context, but the model was optimized for predicting masked words.
-   If we used [MASK] 90% of the time and random words 10% of the time, this would teach the model that the observed word is  _never_  correct.
-   If we used [MASK] 90% of the time and kept the same word 10% of the time, then the model could just trivially copy the non-contextual embedding.

No ablation was done on the ratios of this approach, and it may have worked better with different ratios. In addition, the model performance wasn’t tested with simply masking 100% of the selected tokens.
细节三：对于任务一，对于在数据中随机选择 15% 的标记，其中80%被换位[mask]，10%不变、10%随机替换其他单词，原因是什么？

**两个缺点：**

1、因为Bert用于下游任务微调时， [MASK] 标记不会出现，它只出现在预训练任务中。这就造成了预训练和微调之间的不匹配，微调不出现[MASK]这个标记，模型好像就没有了着力点、不知从哪入手。所以只将80%的替换为[mask]，但这也**只是缓解、不能解决**。

2、相较于传统语言模型，Bert的每批次训练数据中只有 15% 的标记被预测，这导致模型需要更多的训练步骤来收敛。
- NSP
### 特殊符号
- [CLS] 用于分类任务
- [SEP] 用于分割语句
- 

### optimizer
[https://towardsdatascience.com/an-intuitive-understanding-of-the-lamb-optimizer-46f8c0ae4866](https://towardsdatascience.com/an-intuitive-understanding-of-the-lamb-optimizer-46f8c0ae4866)
- size matters
- 

## BERT的fine tune	
- how about [CLS] and [SEP]?
![enter image description here](https://lilianweng.github.io/lil-log/assets/images/BERT-downstream-tasks.png)
- downstream tasks
	- sentence classification(sentiment classification)
	- token classification NER
	- SQuAD & unsupervised SQUAD
- 微调技巧
[https://zhuanlan.zhihu.com/p/109143667](https://zhuanlan.zhihu.com/p/109143667)
	1.  **长文本处理**

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
### task design
- spanBERT [https://zhuanlan.zhihu.com/p/75893972](https://zhuanlan.zhihu.com/p/75893972)
### distillation
### LAMP？not a BERT improvement
## BERT应用
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
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjEwNjI1NjUsLTkwNzk0Mjc5MiwtMjAwNj
M3MTg4NCw4NzQyNDcxODMsLTY4Mzk5MzE2NiwtMzcwMjkyMjM5
LDE3MjMxNDM2NzUsMTQ2NDgxNzkyLDQ0NTMwMzg1OSw2NTU5OD
Y1NzAsLTIwMTk0ODgyMjcsMTE2ODE1Nzg3NywtNDk0MjgxMDk4
LDM1MTI4NDMyLC02MTQxOTc3MjEsLTE5MjI0NjEyMSwxOTU1OD
YzMDc5LC00NzY4NzIyNDUsMTA4NDY2NzgwNSwtNjM4NDQ0ODYy
XX0=
-->