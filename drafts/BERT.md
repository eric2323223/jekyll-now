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
- downstream finetuning

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

- task design (training objective) for self-supervised learning
	- Language model
	- bidirectional LM
	- MLM, NSP
	- GAN (ELATRA)
- The pretrained model is too complex to use
	- distillation
	- 
## Adaptation

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

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3NzI5NDk0OTgsLTgzOTczMjU2MywxND
U4MjAxMjEyLDExMzM2MTEyMjksNzQ3NDQ3ODMyLDExMDU5ODg0
ODgsLTI0NTA1NjQxNywxNTU5OTMzOTQ2LC0xMTY2MzUwNDc2LD
k1NzMyMTQyOCwtMTI4MjQ4NTc0MywtMjE0NzA0MjA4MywtNjY3
Mzg4ODMsLTE2Njg0MzI0OTddfQ==
-->