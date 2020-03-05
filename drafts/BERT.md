# NLP transfer learning 


# Training objective of self-supervised learning - From Word2Vec to Elmo to Bert to XLNet

self-supervised learning is important area because it can greatly reduce the effort of training deep model, 


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
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ3Njg3MjI0NSwxMDg0NjY3ODA1LC02Mz
g0NDQ4NjIsLTc1MzU1OTI3Miw2MDMyMzY2NDIsLTgzOTczMjU2
MywxNDU4MjAxMjEyLDExMzM2MTEyMjksNzQ3NDQ3ODMyLDExMD
U5ODg0ODgsLTI0NTA1NjQxNywxNTU5OTMzOTQ2LC0xMTY2MzUw
NDc2LDk1NzMyMTQyOCwtMTI4MjQ4NTc0MywtMjE0NzA0MjA4My
wtNjY3Mzg4ODMsLTE2Njg0MzI0OTddfQ==
-->