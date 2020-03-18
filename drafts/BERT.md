# NLP transfer learning 

# NLP的迁移学习-BERT篇

# Training objective of self-supervised learning - From Word2Vec to Elmo to Bert to XLNet

self-supervised learning is important area because it can greatly reduce the effort of training deep model, 

作为NLP迁移学习的成功应用，BERT证明了。。。本文旨在介绍BERT模型的结构和设计原理，以及BERT的应用。
## 迁移学习-预训练模型的诞生
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
	- Language model based 
	- 单向 - 双向
	
#### fine tune
- supervised
- unsupervised
- 
## BERT简介
BERT是一个预训练模型，它可以提取输入序列的上下文信息，


- bidirectional
- context dependent embedding
- 

## BERT模型结构
### Transformer encoder based
![enter image description here](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS6Tpws-svHjCwryVCZcEAxIWZ9LjTrg46pSmBG-mi2DMVwDamd)
输出是

### embedding layer
[https://mc.ai/why-bert-has-3-embedding-layers-and-their-implementation-details/](https://mc.ai/why-bert-has-3-embedding-layers-and-their-implementation-details/)
![enter image description here](https://i.stack.imgur.com/QCcYF.png)


## BERT的预训练
### 任务设计
- MLM
[https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
Training the language model in BERT is done by predicting 15% of the tokens in the input, that were randomly picked. These tokens are pre-processed as follows — 80% are replaced with a “[MASK]” token, 10% with a random word, and 10% use the original word. The intuition that led the authors to pick this approach is as follows (Thanks to Jacob Devlin from Google for the insight):

-   If we used [MASK] 100% of the time the model wouldn’t necessarily produce good token representations for non-masked words. The non-masked tokens were still used for context, but the model was optimized for predicting masked words.
-   If we used [MASK] 90% of the time and random words 10% of the time, this would teach the model that the observed word is  _never_  correct.
-   If we used [MASK] 90% of the time and kept the same word 10% of the time, then the model could just trivially copy the non-contextual embedding.

No ablation was done on the ratios of this approach, and it may have worked better with different ratios. In addition, the model performance wasn’t tested with simply masking 100% of the selected tokens.
- NSP
### add special tokens to input
- [CLS] 用于分类任务
- [SEP] 用于分割
- 

### optimizer
[https://towardsdatascience.com/an-intuitive-understanding-of-the-lamb-optimizer-46f8c0ae4866](https://towardsdatascience.com/an-intuitive-understanding-of-the-lamb-optimizer-46f8c0ae4866)
- size matters
- 

## BERT的fine tune	
- how about [CLS] and [SEP]?
- 
- downstream tasks
	- sentence classification(sentiment classification)
	- token classification NER
	- SQUAD & unsupervised SQUAD

## BERT应用
### environment-colab
### DistilBERT


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
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTcyMzE0MzY3NSwxNDY0ODE3OTIsNDQ1Mz
AzODU5LDY1NTk4NjU3MCwtMjAxOTQ4ODIyNywxMTY4MTU3ODc3
LC00OTQyODEwOTgsMzUxMjg0MzIsLTYxNDE5NzcyMSwtMTkyMj
Q2MTIxLDE5NTU4NjMwNzksLTQ3Njg3MjI0NSwxMDg0NjY3ODA1
LC02Mzg0NDQ4NjIsLTc1MzU1OTI3Miw2MDMyMzY2NDIsLTgzOT
czMjU2MywxNDU4MjAxMjEyLDExMzM2MTEyMjksNzQ3NDQ3ODMy
XX0=
-->