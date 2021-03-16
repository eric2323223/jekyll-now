# NLP的迁移学习-GPT篇

## 迁移学习和BERT
> 迁移学习（Transfer Learning）无疑是目前深度学习中的新热点（相对而言）。在计算机视觉领域，它已经应用了一段时间，人们使用经过训练的模型从庞大的ImageNet数据集中学习特征，然后针对较小的数据针对不同的任务对其进行进一步的训练。但是，在NLP中，迁移学习主要限于使用预训练的单词嵌入（这大大改善了基线）。最近，研究人员正在努力将整个模型从一项任务转移到另一项任务，这就是本文的主题。
Sebastian Ruder和Jeremy Howard也许是第一个通过其提出的ULMFiT方法，在NLP中的应用了迁移学习方法，该方法超越了所有最新的文本分类技术。
紧接着，OpenAI 在几个NLP任务上扩大了他们的想法，并超越了SOTA。
在2018年NAACL上，获得最佳论文奖的是介绍ELMo的论文，该论文是一种新的词嵌入技术，与ULMFiT背后的思想非常相似，该技术来自位于UWash的AllenAI和 Luke Zettlemoyer小组的研究人员。
在本文中，我将讨论所有这些新工作以及它们之间的相互关系。让我们从Ruder和Howard的引领潮流的架构开始。

基于注意力机制的transformer取代RNN成为当前主流的网络基础，当前的主流方案主要分为两类方法，第一类只使用transformer的编码器，通过让模型还原在输入中被遮罩的部分来训练模型对输入的理解能力，本质上属于denoise autoencoder，最典型的就是大名鼎鼎的BERT（我们在[NLP迁移学习-BERT篇](https://developer.ibm.com/zh/technologies/machine-learning/articles/nlp-transfer-learning/)中有过比较详尽的介绍）以及一系列对BERT的改进如XLNET。。。；另一类使用了完整的transformer模型，它只是用transformer的解码器。它使用标准的语言模型（Language Model）训练方法进行预训练，本质上属于auto regressive。。。这类方法最典型就是本文的主角-GPT

- auto regressive (GPT)  TEXT generation！！！
- 
## 背景
BERT。。。的时候风光无限，在多个NLP任务中刷新了记录，但是在这份成绩单中缺少了一类重要的NLP任务-文本生成。这是由于BERT的被训练任务被设计为还原少量被屏蔽的词，因此它不善于进行连贯的长序列文本的生成。这却恰恰是GPT的长项，甚至可以这样说，GPT只会做文本生成这一项工作，听起来好像有点弱。。。。
我们已经知道LM作为预训练方法已经有相当长的历史，
作为和BERT同时期的预训练模型，GPT最初的设计目标也是BERT很相似，
- **Learning to Generate Reviews and Discovering Sentiment**， 使用RNN模型，unsupervised 预训练+少量supervised finetuning
> We first trained a  [multiplicative LSTM](https://arxiv.org/abs/1609.07959)  with 4,096 units on a corpus of 82 million Amazon reviews to predict the next character in a chunk of text. Training took one month across four NVIDIA Pascal GPUs, with our model processing 12,500 characters per second.
These 4,096 units (which are just a vector of floats) can be regarded as a feature vector representing the string read by the model. After training the mLSTM, we turned the model into a sentiment classifier by taking a linear combination of these units, learning the weights of the combination via the available supervised data.
While training the linear model with L1 regularization, we noticed it used surprisingly few of the learned units. Digging in, we realized there actually existed a single “sentiment neuron” that’s highly predictive of the sentiment value.
- **Semi-supervised sequence learning**, In contrast to learning a generic representation on one large dataset and then evaluating on other tasks/datasets, Dai & Le (2015) proposed using similar unsupervised objectives such as sequence autoencoding and language modeling to first pretrain a model on a dataset and then finetune it for a given task.

## GPT简介
GPT的全称是Generative Pretraining，它最迟是由openAI在2018年在Improving Language Understanding by Generative Pre-Training的论文中发布的基于Transformer的预训练模型。
GPT的作者之前的一些研究中发现了大量的文本训练可以让基于RNN模型学习到一些可辩别的特征（discriminative features）。 为了验证这些学习到的特征具有通用性，并且可以应用于多种微调任务，作者设计了这样的迁移学习场景：首先对基于Transformer模型应用大量预料进行自监督训练， 然后将预训练模型进行微调后应用到多个微调任务场景（与预训练无关）测试中利用很少量的数据进行监督式训练，最后实验结果在多项公开测试中获得了SOTA，证明了预训练模型具备大量可用于不同领域的知识（如果预训练模型的知识不具备通用性，那么很少量的微调训练数据根本不可能获得好的测试结果）。
> 在本文中，我们探索了一种使用语言监督的半监督方法来进行语言理解任务
无监督的预训练和有监督的微调相结合。 我们的目标是学习通用
几乎不适应各种各样任务的转移。 我们假设可以访问
大量未标记文本和带有手动注释训练示例的几个数据集
（目标任务）。 我们的设置不需要这些目标任务与未标记的任务在同一域中
语料库。 我们采用两阶段的培训程序。 首先，我们在
未标记的数据以学习神经网络模型的初始参数。 随后，我们适应
这些参数用于使用相应的受监督目标的目标任务。


GPT在其诞生和发展之路中逐渐明晰了*** 这种迁移学习的思路比BERT走的更远了一步，它希望预训练模型可以直接用于微调任务，而不需为微调任务设计专门的微调层。


### GPT model
如下图（蓝色方框内）所示，GPT模型基本上就是Transformer模型的解码器部分，不同之处仅仅在于型的输入不同，Transformer解码器的输入是由编码器生成的**句向量**和编码器输出，解码器在解码时要先对已生成的输出进行自注意力计算，再进行编码器-解码器注意力计算；而GPT的输入只有*数字化*（tokenized）用户输入，因此只对用户输入进行自注意力计算。
![enter image description here](https://docs.google.com/drawings/d/e/2PACX-1vTpyx0DRdTPtHBXJKKjRx-ufa_eKX8Cis6fKI99HKVhhwAWKXTT1klDz1ZKm0_VDjjC7-mtVk8YEkqN/pub?w=601&h=411)
GTP预训练模型只使用Transformer 解码器（decoder），在位置编码使用了绝对位置编码，

> Our model largely follows the original transformer work [62]. We trained a 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads). For the position-wise feed-forward networks, we used 3072 dimensional inner states. We used the Adam optimization scheme [27] with a max learning rate of 2.5e-4. The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule. We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens.
> 
GPT模型沿用了transformer 解码器的模型结构，解码器具有12个解码层，每个解码层有12个多头自注意力计算单元（由于没有context vector，去掉了编码器-解码器注意力计算单元），

#### 编码层
- 词编码
GPT解码层使用长度为$h$ 768的浮点数向量来表示每个词，因此需要将tokenzied的词向量$t$（长度为$v$)的长度转换为768位 ， 负责这个转换就是词编码。GPT中采用一个$v*h$转换矩阵 $W_{we}$来实现
$$H  = T * W_{we}$$

`self.wte = nn.Embedding(config.vocab_size, config.n_embd)`
- 位置编码
由于注意力机制不区分词的前后顺序，没有位置的概念，因此需要在将词输入GPT解码层计算之前加入位置信息，同样需要将长度信息转换为GPT解码层可处理的长度$h$,
$$H = T * W_{pe}$$

absolute position embedding `self.wpe = nn.Embedding(config.n_positions, config.n_embd)`
#### GPT解码器
- decoder: language model
![enter image description here](https://qjjnh3a9hpo1nukrg1fwoh71-wpengine.netdna-ssl.com/wp-content/uploads/2019/04/OpenAI-GPT-transformer-decoder_web.jpg)
![enter image description here](https://jalammar.github.io/images/gpt2/self-attention-and-masked-self-attention.png)


### self-supervised Pretrain: 
GPT使用传统的语言模型（Lanaguage model）训练方法
$$h_0 = UW_e + W_p$$
$$h_l = \mathrm{transformerblock}(h_{l-1})  $$
$$P(u) = \mathrm{softmax} (h_nW_e^T)$$
#### Training data
>We use the BooksCorpus dataset [71] for training the language model. It contains over 7,000 unique unpublished books from a variety of genres including Adventure, Fantasy, and Romance. Crucially, it contains long stretches of contiguous text, which allows the generative model to learn to condition on long-range information.
>我们使用BooksCorpus数据集[71]训练语言模型。 它包含7,000多种不同类型的未出版未出版书籍，包括冒险，幻想和浪漫。 至关重要的是，它包含长段连续的文本，这使生成模型可以学习以远程信息为条件。
- attention mask
- GPT is trained on the standard task: given a sequence of prior words, predict the next word.
-  loss function: standard LM
#### Tokenizer
**Byte Pair Encoding** ([**BPE**](https://arxiv.org/abs/1508.07909)) is used to encode the input sequences. BPE was originally proposed as a data compression algorithm in 1990s and then was adopted to solve the open-vocabulary issue in machine translation, as we can easily run into rare and unknown words when translating into a new language. Motivated by the intuition that rare and unknown words can often be decomposed into multiple subwords, BPE finds the best word segmentation by iteratively and greedily merging frequent pairs of characters.
#### 
#### 分类器
dense + softmax
Top K
#### 预训练流程
![enter image description here]( )
### supervised finetune
GPT设计了4种heads处理不同任务： LMhead, (ClfHead，multichoiceHead，similarityHead, inferenceHead, classificationHead)
$$P(y|x^1, ...,x^m) = \mathrm{softmax} (h_l^mW_y)$$
$$L_2(C)=\sum_{(x,y)}\log P(y|x_1, ...,x_m)$$
$$L_3(C)=L_2(C) + \lambda * L_1(C)$$
> - Natural Language Inference: We evaluate on five datasets with diverse sources, including image captions (SNLI), transcribed speech, popular fiction, and government reports (MNLI), Wikipedia articles (QNLI), science exams (SciTail) or news articles (RTE)
> - Question answering and commonsense reasoning: We use the recently released RACE dataset [30], consisting of English passages with associated questions from middle and high school exams
> - Semantic Similarity: Semantic similarity (or paraphrase detection) tasks involve predicting whether two sentences are semantically equivalent or not. The challenges lie in recognizing rephrasing of concepts, understanding negation, and handling syntactic ambiguity. We use three datasets for this task – the Microsoft Paraphrase corpus (MRPC) [14] (collected from news sources), the Quora Question Pairs (QQP) dataset [9], and the Semantic Textual Similarity benchmark (STS-B) [6].
> - Classification: Finally, we also evaluate on two different text classification tasks. The Corpus of Linguistic Acceptability (CoLA) [65] contains expert judgements on whether a sentence is grammatical or not, and tests the innate linguistic bias of trained models. The Stanford Sentiment Treebank (SST-2) [54], on the other hand, is a standard binary classification task.


> The most substantial upgrade that OpenAI GPT proposed is to get rid of the task-specific model and use the pre-trained language model directly!
Let’s take classification as an example. Say, in the labeled dataset, each input has  nn  tokens,  x=(x1,…,xn)x=(x1,…,xn), and one label  yy. GPT first processes the input sequence  xx  through the pre-trained transformer decoder and the last layer output for the last token  xnxn  is  h(n)LhL(n). Then with only one new trainable weight matrix  WyWy, it can predict a distribution over class labels.
![GPT classification](https://lilianweng.github.io/lil-log/assets/images/GPT-classification.png)
P(y∣x1,…,xn)=softmax(h(n)LWy)P(y∣x1,…,xn)=softmax(hL(n)Wy)
The loss is to minimize the negative log-likelihood for true labels. In addition, adding the LM loss as an auxiliary loss is found to be beneficial, because:
-   (1) it helps accelerate convergence during training and
-   (2) it is expected to improve the generalization of the supervised model.
LclsLLML=∑(x,y)∈DlogP(y∣x1,…,xn)=∑(x,y)∈Dlogsoftmax(h(n)L(x)Wy)=−∑ilogp(xi∣xi−k,…,xi−1)=Lcls+λLLMLcls=∑(x,y)∈Dlog⁡P(y∣x1,…,xn)=∑(x,y)∈Dlog⁡softmax(hL(n)(x)Wy)LLM=−∑ilog⁡p(xi∣xi−k,…,xi−1)L=Lcls+λLLM
>With similar designs, no customized model structure is needed for other end tasks (see Fig. 7). If the task input contains multiple sentences, a special delimiter token (`$`) is added between each pair of sentences. The embedding for this delimiter token is a new parameter we need to learn, but it should be pretty minimal.

GPT(GPT1) train different linear layer for specific tasks, such as similarity and multiple choice.
![enter image description here](https://s3.amazonaws.com/clearvoice-media/asg_Dl6YY4jDjq8NXQEH/art_6fF6mdHcMP5MGJeL/1596183183154-1596183183154.png)
![enter image description here](https://qjjnh3a9hpo1nukrg1fwoh71-wpengine.netdna-ssl.com/wp-content/uploads/2019/04/GPT-downstream-tasks_web.jpg)
> **Task-specific input transformations** For some tasks, like text classification, we can directly fine-tune our model as described above. Certain other tasks, like question answering or textual entailment, have structured inputs such as ordered sentence pairs, or triplets of document, question, and answers. Since our pre-trained model was trained on contiguous sequences of text, we require some modifications to apply it to these tasks. Previous work proposed learning task specific architectures on top of transferred representations [44]. Such an approach re-introduces a significant amount of task-specific customization and does not use transfer learning for these additional architectural components. Instead, we use a traversal-style approach [52], where we convert structured inputs into an ordered sequence that our pre-trained model can process. These input transformations allow us to avoid making extensive changes to the architecture across tasks. We provide a brief description of these input transformations below and Figure 1 provides a visual illustration. All transformations include adding randomly initialized start and end tokens ($\langle s \rangle, \langle e\rangle$). 
> **Textual entailment** For entailment tasks, we concatenate the premise $p$ and hypothesis $h$ token sequences, with a delimiter token ($) in between. 
> **Similarity** For similarity tasks, there is no inherent ordering of the two sentences being compared. To reflect this, we modify the input sequence to contain both possible sentence orderings (with a delimiter in between) and process each independently to produce two sequence representations $h^m_l$which are added element-wise before being fed into the linear output layer. 
> **Question Answering and Commonsense Reasoning** For these tasks, we are given a context document z, a question q, and a set of possible answers {$a_k$}. We concatenate the document context and question with each possible answer, adding a delimiter token in between to get [z; q; $; ak]. Each of these sequences are processed independently with our model and then normalized via a softmax layer to produce an output distribution over possible answers.

### 实验数据分析
GPT论文在GPT模型上做了大量实验，并根据实验结果提出了一些观点，这些观点也影响了GPT后续的发展方向
![enter image description here](https://d3i71xaburhd42.cloudfront.net/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035/7-Figure2-1.png)
- 模型深度对迁移学习的影响
GPT作者训练了多种不同深度（GPT解码器层数）的预训练模型，并在微调任务（RACE和MultiNLI）中对不同预训练模型的预测结果比较，如上图1中所示，可以观察到随着模型深度（层数）的增加，预测准确率也随之上升。这个结果说明了增加解码器的层数可以让预训练模型携带更多可重用的特征，帮助提高微调任务的效果
- 零次学习（zero shot）
在另一个实验中，作者尝试了零次学习，也就是使用经过不同训练量训练出来的多种预训练模型，在不经过微调训练而直接在微调任务（sentiment analysis...）的场景中进行预测，如上图2所示。实验结果显示预训练量的提高能普遍提高零次学习在各个微调任务中的准确率。这提示了使用更大的数据集进行预训练可以提高。。。。另外比较图中LSTM和Transformer的曲线也进一步印证了Transformer的。。。

## GPT设计思想
GPT设计思想的诞生可以追述到
“representation learning”

### GPT1: 可以直接加速finetune训练
![enter image description here](https://openai.com/content/images/2018/06/zero-shot-transfer@2x.png)


### the power of scale
![enter image description here](https://miro.medium.com/max/625/1*q-P5aQ7A6VlsfroP3ckg8A.jpeg)
![enter image description here](https://bmk.sh/images/gpt3/perf-small.png)


GPT3 already have most of the knowlege you can think of, the key is how to let GPT3 understand the task.
GPT3/GPT2 are not strictly in transfer learning scope, because they don't need finetune.

### GTP vs BERT

> we noted that unsupervised learning techniques can yield surprisingly discriminative features when trained on enough data.Here, we wanted to further explore this idea: can we develop one model, train it in an unsupervised way on a large amount of data, and then fine-tune the model to achieve good performance on many different tasks? Our results indicate that this approach works surprisingly well; the same core model can be fine-tuned for very different tasks with minimal adaptation.
> 我们注意到，无监督学习技术在训练足够多的数据时可以产生出乎意料的区别特征。在这里，我们想进一步探索这一思想：我们可以开发一个模型，以无监督的方式对大量数据进行训练，然后精细地- 调整模型以在许多不同的任务上实现良好的性能？ 我们的结果表明，这种方法行之有效。 可以以最小的适应性为不同的任务微调相同的核心模型。
-   GPT-2 and BERT at the two leading language models out there at time of writing in early 2020. They are the same in that they are both based on the transformer architecture, but they are fundamentally different in that BERT has just the  _encoder_  blocks from the transformer, whilst GPT-2 has just the  _decoder_  blocks from the transformer.

### GPT2/ GPT3
GPT-3 demonstrates that a language model trained on enough data can solve NLP tasks that it has never encountered. That is, GPT-3 studies the model as a general solution for many downstream jobs  **without fine-tuning**.

### **BERT vs GPT-3 — The Right Comparison**
Both the models —  [GPT-3](https://analyticsindiamag.com/how-openais-gpt-3-can-be-alarming-for-the-society/)  and  [BERT](https://analyticsindiamag.com/bert-classifier-with-tensorflow-2-0/)  have been relatively new for the industry, but their state-of-the-art performance has made them the winners among other models in the natural language processing field. However, being trained on 175 billion parameters,  [GPT-3](https://analyticsindiamag.com/5-jobs-that-gpt-3-might-challenge/)  becomes 470 times bigger in size than BERT-Large.

Secondly, while  [BERT](https://analyticsindiamag.com/step-by-step-guide-to-implement-multi-class-classification-with-bert-tensorflow/)  requires an elaborated fine-tuning process where users have to gather data of examples to train the model for specific downstream tasks, GPT-3’s text-in and text-out API allows the users to reprogram it using instructions and access it. Case in point — for sentiment analysis or question answering tasks, to use BERT, the users have to train the model on a separate layer on sentence encodings. However,  [GPT-3](https://analyticsindiamag.com/gpt-3-has-weaknesses-and-makes-silly-mistakes-sam-altman-openai/)  uses a few-shot learning process on the input token to predict the output result.
###  **[Text Generation with a Language Model](https://code.oursky.com/ai-text-generator-text-generation-with-a-gpt2-model/)**

- greedy search
- Beam search
- Pure Sampling
- Top-k Sampling and Sampling with Temperature
- Nucleus Sampling
## GPT的发展 
### GPT-1: Improving Language Understanding by Generative Pre-Training
> We demonstrate that large gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. In contrast to previous approaches, we make use of task-aware input transformations during fine-tuning to achieve effective transfer while requiring minimal changes to the model architecture. We demonstrate the effectiveness of our approach on a wide range of benchmarks for natural language understanding
> 们证明，通过在各种未标记文本的语料库上对语言模型进行生成式预训练，然后对每个特定任务进行区分性微调，可以实现这些任务的巨大收益。 与以前的方法相比，我们在微调过程中利用了任务感知的输入转换来实现有效的传递，同时对模型体系结构的更改要求最小。 我们在广泛的自然语言理解基准测试中证明了我们的方法的有效性

### GPT-2: Language Models are Unsupervised Multitask Learners
相比GPT有两点变化
- zero-shot
- LM is all you need
#### LM works with all types of finetune tasks
The most substantial upgrade that OpenAI GPT proposed is to get rid of the task-specific model and use the pre-trained language model directly!
-   Language modeling is a very difficult task, even for humans.
-   Language models are expected to compress any possible context into a vector that generalizes over possible completions.
	 -   “They walked down the street to ???”
 -   To have any chance at solving this task, a model is forced to learn syntax, semantics, encode facts about the world, etc.
-   Given enough data, a huge model, and enough compute, can do a reasonable job!
-   Empirically works better than translation, autoencoding: “Language Modeling Teaches You More Syntax than Translation Does”
![enter image description here](https://joeddav.github.io/blog/images/zsl/gpt3_triviahq.png)
GPT2的创新点在于验证了无监督的语言建模能够学习到有监督任务所需的特征。原文是
> We demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText.
> 我们证明，当在名为WebText的数百万个网页的新数据集上进行训练时，语言模型开始在没有任何明确监督的情况下开始学习这些任务。

这个才是GPT-2文章价值所在。

>GPT-2 displays a broad set of capabilities, including the ability to generate conditional synthetic text samples of unprecedented quality, where we prime the model with an input and have it generate a lengthy continuation. In addition, GPT-2 outperforms other language models trained on specific domains (like Wikipedia, news, or books) without needing to use these domain-specific training datasets. On language tasks like question answering, reading comprehension, summarization, and translation, GPT-2 begins to learn these tasks from the raw text, using no task-specific training data. While scores on these downstream tasks are far from state-of-the-art, they suggest that the tasks can benefit from unsupervised techniques, given sufficient (unlabeled) data and compute.
>GPT-2显示了广泛的功能，包括生成具有空前质量的条件合成文本样本的能力，我们在模型中使用输入来填充模型并让其生成冗长的延续。 此外，GPT-2优于在特定领域（如Wikipedia，新闻或书籍）上训练的其他语言模型，而无需使用这些特定于领域的训练数据集。 在诸如答疑，阅读理解，总结和翻译之类的语言任务上，GPT-2开始使用原始文本来学习这些任务，而没有使用特定于任务的训练数据。 尽管这些下游任务的得分远非最新水平，但它们表明，只要有足够的（未标记）数据和计算，这些任务就可以从无监督的技术中受益。
![enter image description here](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQUFL2kZa4Nzhvte_2-qmBsFtnoCQ_6ffJvDg&usqp=CAU)
> In this paper, we connect these two lines of work and continue the trend of more general methods of transfer. We demonstrate language models can perform down-stream tasks in a zero-shot setting – without any parameter or architecture modification. We demonstrate this approach shows potential by highlighting the ability of language models to perform a wide range of tasks in a zero-shot setting. We achieve promising, competitive, and state of the art results depending on the task.
> 在本文中，我们将这两方面的工作联系起来，并延续了更通用的转移方法的趋势。 我们演示了语言模型可以在零触发设置下执行下游任务-无需任何参数或体系结构修改。 我们通过强调语言模型在零镜头设置下执行各种任务的能力来证明这种方法显示出了潜力。 我们根据任务获得有希望的，有竞争力的和最先进的结果。
#### training data
> The resulting dataset, WebText, contains the text subset of these 45 million links. To extract the text from HTML responses we use a combination of the Dragnet (Peters & Lecocq, 2013) and Newspaper1 content extractors. All results presented in this paper use a preliminary version of WebText which does not include links created after Dec 2017 and which after de-duplication and some heuristic based cleaning contains slightly over 8 million documents for a total of 40 GB of text. We removed all Wikipedia documents from WebText since it is a common data source for other datasets and could complicate analysis due to over1https://github.com/codelucas/newspaper Language Models are Unsupervised Multitask Learners lapping training data with test evaluation tasks.  
> 结果数据集WebText包含这4500万个链接的文本子集。 为了从HTML响应中提取文本，我们使用了Dragnet（Peters和Lecocq，2013）和Newspaper1内容提取器的组合。 本文介绍的所有结果均使用WebText的初步版本，该版本不包含2017年12月之后创建的链接，该链接在重复数据删除和基于启发式的清理后包含略超过800万份文档，总计40 GB文本。 我们从WebText中删除了所有Wikipedia文档，因为它是其他数据集的通用数据源，并且由于过于复杂而导致分析复杂化。
#### model
> The model largely follows the details of the OpenAI GPT model (Radford et al., 2018) with a Parameters Layers dmodel 117M 12 768 345M 24 1024 762M 36 1280 1542M 48 1600 Table 2. Architecture hyperparameters for the 4 model sizes. few modifications. Layer normalization (Ba et al., 2016) was moved to the input of each sub-block, similar to a pre-activation residual network (He et al., 2016) and an additional layer normalization was added after the final selfattention block. A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of 1/ √ N where N is the number of residual layers. The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens and a larger batchsize of 512 is used.
#### Experiments
> We trained and benchmarked four LMs with approximately log-uniformly spaced sizes. The architectures are summarized in Table 2. The smallest model is equivalent to the original GPT, and the second smallest equivalent to the largest model from BERT (Devlin et al., 2018). Our largest model, which we call GPT-2, has over an order of magnitude more parameters than GPT. The learning rate of each model was manually tuned for the best perplexity on a 5% held-out sample of WebText. All models still underfit WebText and held-out perplexity has as of yet improved given more training time.

 |Parameters |Layers |dmodel|
 |---|---|---|---|
 |117M |12 |768|
 |345M |24 |1024|
 |762M |36 |1280|
 |1542M |48 |1600| 
Table 2. Architecture hyperparameters for the 4 model sizes.
#### Analysis
> These findings suggest a promising path towards building language processing systems which learn to perform tasks from their naturally occurring demonstrations.
> 这些发现为建立语言处理系统提供了一条有希望的途径，该系统将从自然发生的演示中学习执行任务。
#### Conclusion
> When a large language model is trained on a sufficiently large and diverse dataset it is able to perform well across many domains and datasets. GPT-2 zero-shots to state of the art performance on 7 out of 8 tested language modeling datasets. The diversity of tasks the model is able to perform in a zero-shot setting suggests that high-capacity models trained to maximize the likelihood of a sufficiently varied text corpus begin to learn how to perform a surprising amount of tasks without the need for explicit supervision.
> 在足够大且多样化的数据集上训练大型语言模型时，它能够在许多域和数据集上表现良好。 GPT-2对8个经过测试的语言建模数据集中的7个进行了最新性能的零射。 该模型能够在零镜头设置下执行的任务的多样性表明，经过训练以使文本语料库充分变化的可能性最大化的高容量模型开始学习如何执行数量惊人的任务，而无需明确的监督 。
### GPT-3: Language Models are Few-Shot Learners
**可以跳过finetune训练直接使用**


-   GPT-3 showcases how a language model trained on a massive range of data can solve various NLP tasks without fine-tuning.

GPT-3依旧延续自己的单向语言模型训练方式，只不过这次把模型尺寸增大到了**1750亿**，并且使用**45TB**数据进行训练。同时，GPT-3主要聚焦于更通用的NLP模型，解决当前BERT类模型的两个缺点：

1.  **对领域内有标签数据的过分依赖**：虽然有了预训练+精调的两段式框架，但还是少不了一定量的领域标注数据，否则很难取得不错的效果，而标注数据的成本又是很高的。
2.  **对于领域数据分布的过拟合**：在精调阶段，因为领域数据有限，模型只能拟合训练数据分布，如果数据较少的话就可能造成过拟合，致使模型的泛华能力下降，更加无法应用到其他领域。

因此GPT-3的主要目标是**用更少的领域数据、且不经过精调步骤去解决问题**。

- zero shot learning
- one shot learning
- few shot learning
![enter image description here](https://miro.medium.com/max/448/1*2dX-PZSNdmj0KOa-NmjrEA.jpeg)
> Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art finetuning approaches.


Question？Improve scale make real intelligent？ YES?NO? GPT 不会做多位数的加减法。
GPT3并没有在理论上进行任何创新（仅仅是一次费用高昂的实验报告），它的价值更多体现在实际的应用中，

  #### facts
- 2048 word vector
- 96 transformer layers
- 96 self-attention heads, each 128 dimensional
- 12288 units in bottleneck layer, 49152 in feed forward layer
- batch size of 3.2M samples

#### pretraining
- trained on 499 Billion tokens
- Would require 355 years and $4600000 train on cheapest GPU cloud

![enter image description here](https://miro.medium.com/max/4344/1*l8h-W_Y3atnWUVYyQL06jQ.png)

## look ahead
> 对于GPT-3而言，它最大的价值是在无监督下的自我学习能力，以及纯粹通过扩大规模实现性能提升。后者已经在GPT-3的论文中得到验证，数据越大，参数量越大，模型的性能表现越好。 其实，GPT-3与GPT-2本质上差异并不大，只是在数据量和参数量两个方面扩大了100倍，便得到了远超GPT-2的性能。
长远来看，我们唯一可以确定的是，未来我们会创造越来越多的数据和计算能力，那么，它将意味着GPT-3的迭代版会越来越强。至于未来GPT-3会达到怎样的程度，深度学习之父、图灵奖得主Hinton称，" 如以GPT3惊人性能预测，未来生命，宇宙和万物的答案也不过是4.398万亿个参数而已。”
## 总结

gpt-3 is a huge look-up table
- it cannot understand basic numaric calculation
- 

然而，**GPT-3还是存在一些局限，论文作者给出了未来有前景的方向：**

-   建立GPT-3尺度的双向模型。
-   使双向模型能在少样本、零样本学习上工作。


## reference
[gpt2 and bert](https://www.kaggle.com/residentmario/notes-on-gpt-2-and-bert-models)
[illustrated gpt2](http://jalammar.github.io/illustrated-gpt2/)
[gpt3 visualized](http://jalammar.github.io/how-gpt3-works-visualizations-animations/)
[openai gpt3: lanaguage models are few-shot learners](https://medium.com/analytics-vidhya/openai-gpt-3-language-models-are-few-shot-learners-82531b3d3122)
[PET](https://analyticsindiamag.com/can-this-tiny-language-model-defeat-gigantic-gpt3/)
[GPT2 and BERT a comparison](https://judithvanstegeren.com/blog/2020/GPT2-and-BERT-a-comparison.html)
[Transfer learning in NLP](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5888218f39_1_161)
[OpenAI GPT3 LM](https://www.slideshare.net/numenta/openais-gpt-3-language-model-guest-steve-omohundro)
[implications of gpt3](https://www.slideshare.net/RavenJiang/implications-of-gpt3)
[GPT3 a brief summary](https://bmk.sh/2020/05/29/GPT-3-A-Brief-Summary/)
[generalized language model](https://www.topbots.com/generalized-language-models-ulmfit-openai-gpt/)
[autoCoder](https://wangcongcong123.github.io/AutoCoder/)
[GPT-3 primer](https://towardsdatascience.com/gpt-3-primer-67bc2d821a00)
[Scaling Laws for Neural Language models](https://arxiv.org/pdf/2001.08361.pdf)
[openai gpt2原理解读](https://zhuanlan.zhihu.com/p/57251615)
[GPT-3: A Hitchhiker's Guide](https://lambdalabs.com/blog/gpt-3/)
[OpenAI's GPT-3 Language Model: A Technical Overview](https://lambdalabs.com/blog/demystifying-gpt-3/)
[generalized language models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#openai-gpt)
[Language models](https://docs.google.com/presentation/d/1sdH-9KQipnu3RMN0-YUqU8R4ZMLapz8IJzfe7VLN39o/edit#slide=id.p)
[gpt3-language-models-are-few-shot-learners](https://blog.inten.to/gpt-3-language-models-are-few-shot-learners-a13d1ae8b1f9)
[NLP模型应用之三：GPT与GPT-2](https://www.jianshu.com/p/1571bfe0af01)
[Improving Language Understanding by Generative Pre-Training](https://www.cs.princeton.edu/courses/archive/spring20/cos598C/lectures/lec4-pretraining.pdf)
[GPT3-the-database-prompt](https://www.gwern.net/GPT-3#the-database-prompt)
[what can you do with the openai gpt-3](https://blog.exxactcorp.com/what-can-you-do-with-the-openai-gpt-3-language-model/)
[Understanding the GPT-2 source code](https://medium.com/analytics-vidhya/understanding-the-gpt-2-source-code-part-5-87bbe21dd749)
[Generating text summaries with GPT2](https://blog.paperspace.com/generating-text-summaries-gpt-2/)
[Practical applications of GPT2](https://medium.com/the-research-nest/practical-applications-of-open-ais-gpt-2-deep-learning-model-14701f18a432)
[Fine-Tuning GPT-2 from Human Preferences](https://openai.com/blog/fine-tuning-gpt-2/)
[Unsupervised sentiment neuron](https://openai.com/blog/unsupervised-sentiment-neuron/)
[What is GPT3](https://www.rev.com/blog/what-is-gpt-3-the-new-openai-language-model)
[Illustrated GPT2](http://jalammar.github.io/illustrated-gpt2/#part-3-beyond-language-modeling)
[How to Build an AI Text Generator: Text Generation with a GPT-2 Model](https://code.oursky.com/ai-text-generator-text-generation-with-a-gpt2-model/)
[GTP model explained](https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2)


$$P(u) = \underset{x} \mathrm{softmax} (h_nW_e^T)$$
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTU1NTQyODA0LDY0Mjk3NDIzNywxMzY4ND
g5MTA5LDE3MjczNTYwMDAsLTEzNDk2MzA4OTAsLTE3Njc0MjMw
NzIsLTQxMDE5NTQyMSwtMTI4NTc3NzgxNCwxMDMwMzY2ODYsMT
QxNjI1MTE4MiwxOTUwNDU1MzkyLDEzNjAwMDY4NDQsLTM5NzQ5
MTUwOSwxOTQyNjg5NTE4LDEyNjkwNDA2OTMsLTUwMjYwMjgwLD
k4NjkwNzYyMSw1NDM3NTQ0MzcsLTMwMjU4NjUyMiwtMTMyNTgy
NTQ2Ml19
-->