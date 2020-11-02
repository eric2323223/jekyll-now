# NLP的迁移学习-GPT篇

## 迁移学习和BERT
> 迁移学习（Transfer Learning）无疑是目前深度学习中的新热点（相对而言）。在计算机视觉领域，它已经应用了一段时间，人们使用经过训练的模型从庞大的ImageNet数据集中学习特征，然后针对较小的数据针对不同的任务对其进行进一步的训练。但是，在NLP中，迁移学习主要限于使用预训练的单词嵌入（这大大改善了基线）。最近，研究人员正在努力将整个模型从一项任务转移到另一项任务，这就是本文的主题。
Sebastian Ruder和Jeremy Howard也许是第一个通过其提出的ULMFiT方法，在NLP中的应用了迁移学习方法，该方法超越了所有最新的文本分类技术。
紧接着，OpenAI 在几个NLP任务上扩大了他们的想法，并超越了SOTA。
在2018年NAACL上，获得最佳论文奖的是介绍ELMo的论文，该论文是一种新的词嵌入技术，与ULMFiT背后的思想非常相似，该技术来自位于UWash的AllenAI和 Luke Zettlemoyer小组的研究人员。
在本文中，我将讨论所有这些新工作以及它们之间的相互关系。让我们从Ruder和Howard的引领潮流的架构开始。

基于注意力机制的transformer取代RNN成为当前主流的网络基础，当前的主流方案主要分为两类方法，第一类只使用transformer的编码器，通过在输入中引入噪音  训练，本质上属于denoise autoencoder，最典型的就是大名鼎鼎的BERT（我们在**中有过比较详尽的介绍）以及一系列对BERT的改进如XLNET。。。；另一类使用了完整的transformer模型（包括编码器和解码器）

- auto regressive (GPT)  TEXT generation！！！


## GPT简介

### preprocessor 
**Byte Pair Encoding** ([**BPE**](https://arxiv.org/abs/1508.07909)) is used to encode the input sequences. BPE was originally proposed as a data compression algorithm in 1990s and then was adopted to solve the open-vocabulary issue in machine translation, as we can easily run into rare and unknown words when translating into a new language. Motivated by the intuition that rare and unknown words can often be decomposed into multiple subwords, BPE finds the best word segmentation by iteratively and greedily merging frequent pairs of characters.
### GPT model
GTP预训练模型只使用Transformer 解码器（decoder），在位置编码使用了绝对位置编码，
![enter image description here](https://cdn-images-1.medium.com/max/1600/1*Ji79bZ3KqpMAjZ9Txv4q8Q.png)
#### embedding
- token embedding
`self.wte = nn.Embedding(config.vocab_size, config.n_embd)`
- position embedding
absolute position embedding `self.wpe = nn.Embedding(config.n_positions, config.n_embd)`
#### transformer decoder
- decoder: language model
	- top K
![enter image description here](https://qjjnh3a9hpo1nukrg1fwoh71-wpengine.netdna-ssl.com/wp-content/uploads/2019/04/OpenAI-GPT-transformer-decoder_web.jpg)
### Pretrain: 
- attention mask
-   GPT is trained on the standard task: given a sequence of prior words, predict the next word.
-  loss function: standard LM

### finetune
**no model justification!!!**
![enter image description here](https://qjjnh3a9hpo1nukrg1fwoh71-wpengine.netdna-ssl.com/wp-content/uploads/2019/04/GPT-downstream-tasks_web.jpg)
- zero shot learning
- one shot learning
- few shot learning
![enter image description here](https://miro.medium.com/max/448/1*2dX-PZSNdmj0KOa-NmjrEA.jpeg)

## GPT设计思想
### LM is all you need
-   Language modeling is a very difficult task, even for humans.
-   Language models are expected to compress any possible context into a vector that generalizes over possible completions.
	 -   “They walked down the street to ???”
 -   To have any chance at solving this task, a model is forced to learn syntax, semantics, encode facts about the world, etc.
-   Given enough data, a huge model, and enough compute, can do a reasonable job!
-   Empirically works better than translation, autoencoding: “Language Modeling Teaches You More Syntax than Translation Does”
### The most substantial upgrade that OpenAI GPT proposed is to get rid of the task-specific model and use the pre-trained language model directly!
### GPT1: 可以直接加速finetune训练
![enter image description here](https://openai.com/content/images/2018/06/zero-shot-transfer@2x.png)
### LM works with all types of finetune tasks
The most substantial upgrade that OpenAI GPT proposed is to get rid of the task-specific model and use the pre-trained language model directly!
### the power of scale
![enter image description here](https://miro.medium.com/max/625/1*q-P5aQ7A6VlsfroP3ckg8A.jpeg)
![enter image description here](https://bmk.sh/images/gpt3/perf-small.png)


GPT3 already have most of the knowlege you can think of, the key is how to let GPT3 understand the task.
GPT3/GPT2 are not strictly in transfer learning scope, because they don't need finetune.

### GTP vs BERT

> we noted that unsupervised learning techniques can yield surprisingly discriminative features when trained on enough data.Here, we wanted to further explore this idea: can we develop one model, train it in an unsupervised way on a large amount of data, and then fine-tune the model to achieve good performance on many different tasks? Our results indicate that this approach works surprisingly well; the same core model can be fine-tuned for very different tasks with minimal adaptation.

-   GPT-2 and BERT at the two leading language models out there at time of writing in early 2020. They are the same in that they are both based on the transformer architecture, but they are fundamentally different in that BERT has just the  _encoder_  blocks from the transformer, whilst GPT-2 has just the  _decoder_  blocks from the transformer.

### GPT2/ GPT3
GPT-3 demonstrates that a language model trained on enough data can solve NLP tasks that it has never encountered. That is, GPT-3 studies the model as a general solution for many downstream jobs  **without fine-tuning**.

### **BERT vs GPT-3 — The Right Comparison**
Both the models —  [GPT-3](https://analyticsindiamag.com/how-openais-gpt-3-can-be-alarming-for-the-society/)  and  [BERT](https://analyticsindiamag.com/bert-classifier-with-tensorflow-2-0/)  have been relatively new for the industry, but their state-of-the-art performance has made them the winners among other models in the natural language processing field. However, being trained on 175 billion parameters,  [GPT-3](https://analyticsindiamag.com/5-jobs-that-gpt-3-might-challenge/)  becomes 470 times bigger in size than BERT-Large.

Secondly, while  [BERT](https://analyticsindiamag.com/step-by-step-guide-to-implement-multi-class-classification-with-bert-tensorflow/)  requires an elaborated fine-tuning process where users have to gather data of examples to train the model for specific downstream tasks, GPT-3’s text-in and text-out API allows the users to reprogram it using instructions and access it. Case in point — for sentiment analysis or question answering tasks, to use BERT, the users have to train the model on a separate layer on sentence encodings. However,  [GPT-3](https://analyticsindiamag.com/gpt-3-has-weaknesses-and-makes-silly-mistakes-sam-altman-openai/)  uses a few-shot learning process on the input token to predict the output result.
### GPT-1: Improving Language Understanding by Generative Pre-Training
### GPT-2: Language Models are Unsupervised Multitask Learners
GPT2的创新点在于验证了无监督的语言建模能够学习到有监督任务所需的特征。原文是
> We demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText.

这个才是GPT-2文章价值所在。


>GPT-2 displays a broad set of capabilities, including the ability to generate conditional synthetic text samples of unprecedented quality, where we prime the model with an input and have it generate a lengthy continuation. In addition, GPT-2 outperforms other language models trained on specific domains (like Wikipedia, news, or books) without needing to use these domain-specific training datasets. On language tasks like question answering, reading comprehension, summarization, and translation, GPT-2 begins to learn these tasks from the raw text, using no task-specific training data. While scores on these downstream tasks are far from state-of-the-art, they suggest that the tasks can benefit from unsupervised techniques, given sufficient (unlabeled) data and compute.
   
### GPT-3: Language Models are Few-Shot Learners
**可以跳过finetune训练直接使用**


-   GPT-3 showcases how a language model trained on a massive range of data can solve various NLP tasks without fine-tuning.

GPT-3依旧延续自己的单向语言模型训练方式，只不过这次把模型尺寸增大到了**1750亿**，并且使用**45TB**数据进行训练。同时，GPT-3主要聚焦于更通用的NLP模型，解决当前BERT类模型的两个缺点：

1.  **对领域内有标签数据的过分依赖**：虽然有了预训练+精调的两段式框架，但还是少不了一定量的领域标注数据，否则很难取得不错的效果，而标注数据的成本又是很高的。
2.  **对于领域数据分布的过拟合**：在精调阶段，因为领域数据有限，模型只能拟合训练数据分布，如果数据较少的话就可能造成过拟合，致使模型的泛华能力下降，更加无法应用到其他领域。

因此GPT-3的主要目标是**用更少的领域数据、且不经过精调步骤去解决问题**。

Question？Improve scale make real intelligent？ NO！GPT 不会做多位数的加减法。
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
<!--stackedit_data:
eyJoaXN0b3J5IjpbOTY3MDA5NDg5LDE5Njg5OTY2NjksLTEzNj
MzNTU2NjksLTExODA4NDMxOTMsNDc4NDcxODY2LDEyNjQ1MTU5
MDgsLTc2NDMxNTA3NCwtODc4NjcwMjAwLC00OTE3MzkzMTUsMT
gwOTQ2MzQxOCwtMTU2ODUyMTc4MCw5NDI3NjgwNDAsODU1NDQz
MjQ0LDEzMDk0NzU1MzksMTgyOTY2MzQ5NCwtMTYwMDU5NzMyOS
w4NDIzNDMyMjIsLTM2NTQwNzg5NiwtMTU3NTQ1MDkzOCwtNTcy
MTQ2NjE2XX0=
-->