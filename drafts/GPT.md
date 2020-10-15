# NLP的迁移学习-GPT篇

## 迁移学习和BERT
- auto regressive (GPT)
- denoise autoencoder (BERT)

## GPT简介


### preprocessor 
**Byte Pair Encoding** ([**BPE**](https://arxiv.org/abs/1508.07909)) is used to encode the input sequences. BPE was originally proposed as a data compression algorithm in 1990s and then was adopted to solve the open-vocabulary issue in machine translation, as we can easily run into rare and unknown words when translating into a new language. Motivated by the intuition that rare and unknown words can often be decomposed into multiple subwords, BPE finds the best word segmentation by iteratively and greedily merging frequent pairs of characters.
### GPT model
![enter image description here](https://cdn-images-1.medium.com/max/1600/1*Ji79bZ3KqpMAjZ9Txv4q8Q.png)
#### embedding
- token embedding + position embedding
#### transformer
- encoder: masked self-attention
- decoder: language model
	- top K
![enter image description here](https://qjjnh3a9hpo1nukrg1fwoh71-wpengine.netdna-ssl.com/wp-content/uploads/2019/04/OpenAI-GPT-transformer-decoder_web.jpg)
### Pretrain: 


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
### LM works with all types of finetune tasks
The most substantial upgrade that OpenAI GPT proposed is to get rid of the task-specific model and use the pre-trained language model directly!
### the power of scale
![enter image description here](https://miro.medium.com/max/625/1*q-P5aQ7A6VlsfroP3ckg8A.jpeg)
![enter image description here](https://bmk.sh/images/gpt3/perf-small.png)


GPT3 already have most of the knowlege you can think of, the key is how to let GPT3 understand the task.
GPT3/GPT2 are not strictly in transfer learning scope, because they don't need finetune.

### GTP vs BERT
-   GPT-2 and BERT at the two leading language models out there at time of writing in early 2020. They are the same in that they are both based on the transformer architecture, but they are fundamentally different in that BERT has just the  _encoder_  blocks from the transformer, whilst GPT-2 has just the  _decoder_  blocks from the transformer.

### GPT2/ GPT3
GPT-3 demonstrates that a language model trained on enough data can solve NLP tasks that it has never encountered. That is, GPT-3 studies the model as a general solution for many downstream jobs  **without fine-tuning**.

### **BERT vs GPT-3 — The Right Comparison**
Both the models —  [GPT-3](https://analyticsindiamag.com/how-openais-gpt-3-can-be-alarming-for-the-society/)  and  [BERT](https://analyticsindiamag.com/bert-classifier-with-tensorflow-2-0/)  have been relatively new for the industry, but their state-of-the-art performance has made them the winners among other models in the natural language processing field. However, being trained on 175 billion parameters,  [GPT-3](https://analyticsindiamag.com/5-jobs-that-gpt-3-might-challenge/)  becomes 470 times bigger in size than BERT-Large.

Secondly, while  [BERT](https://analyticsindiamag.com/step-by-step-guide-to-implement-multi-class-classification-with-bert-tensorflow/)  requires an elaborated fine-tuning process where users have to gather data of examples to train the model for specific downstream tasks, GPT-3’s text-in and text-out API allows the users to reprogram it using instructions and access it. Case in point — for sentiment analysis or question answering tasks, to use BERT, the users have to train the model on a separate layer on sentence encodings. However,  [GPT-3](https://analyticsindiamag.com/gpt-3-has-weaknesses-and-makes-silly-mistakes-sam-altman-openai/)  uses a few-shot learning process on the input token to predict the output result.
### GPT-1: Improving Language Understanding by Generative Pre-Training
### GPT-2: Language Models are Unsupervised Multitask Learners
### GPT-3: Language Models are Few-Shot Learners
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



## reference
[gpt2 and bert](https://www.kaggle.com/residentmario/notes-on-gpt-2-and-bert-models)
[gpt3](https://medium.com/analytics-vidhya/openai-gpt-3-language-models-are-few-shot-learners-82531b3d3122)
[PET](https://analyticsindiamag.com/can-this-tiny-language-model-defeat-gigantic-gpt3/)
[GPT2 and BERT a comparison](https://judithvanstegeren.com/blog/2020/GPT2-and-BERT-a-comparison.html)
[Transfer learning in NLP](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5888218f39_1_161)
[OpenAI GPT3 LM](https://www.slideshare.net/numenta/openais-gpt-3-language-model-guest-steve-omohundro)
[GPT3 a brief summary](https://bmk.sh/2020/05/29/GPT-3-A-Brief-Summary/)
[generalized language model](https://www.topbots.com/generalized-language-models-ulmfit-openai-gpt/)
[autoCoder](https://wangcongcong123.github.io/AutoCoder/)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjAxNTI3NTYzNiwtMTI5NDM4ODY3NCwxNT
YzMzAzMDQ5LDE1Nzg4MzY0OTksLTEwNzUzMTAxMTMsNjE2NjYw
MzY2LDIwMzE1MDMyNDksLTE4NzMzMjY5NTQsLTE5MjE5OTQzNj
UsLTczMjE1MjM5NiwtMTk2Mjc5OTkxNiwzMDc0ODg3NjgsLTkw
MjY3NTQ5OCwtMTc5MDkzNTI2MiwtMjAxNzI0NjIsLTQ1MjY3Mj
E4MiwtMTc0MjIwNzk4NSwxMjAwMzI5MjI5LC05OTY3NzQ2NTYs
OTg0MzQ0NDk3XX0=
-->