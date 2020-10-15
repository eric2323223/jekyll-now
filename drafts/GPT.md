# NLP的迁移学习-GPT篇

## 迁移学习和BERT
- auto regressive (GPT)
- denoise autoencoder (BERT)

## GPT简介


### preprocessor 
- tokenizer = BPE
### GPT model
![enter image description here](https://cdn-images-1.medium.com/max/1600/1*Ji79bZ3KqpMAjZ9Txv4q8Q.png)
#### embedding
- token embedding + position embedding
#### transformer
- encoder: masked self-attention
- decoder: 
## GPT设计思想
### Pretrain: LM is all you need
-   Language modeling is a very difficult task, even for humans.
-   Language models are expected to compress any possible context into a vector that generalizes over possible completions.
	 -   “They walked down the street to ???”
 -   To have any chance at solving this task, a model is forced to learn syntax, semantics, encode facts about the world, etc.
-   Given enough data, a huge model, and enough compute, can do a reasonable job!
-   Empirically works better than translation, autoencoding: “Language Modeling Teaches You More Syntax than Translation Does”

### finetune
- zero shot learning
- one shot learning
- few shot learning
![enter image description here](https://miro.medium.com/max/448/1*2dX-PZSNdmj0KOa-NmjrEA.jpeg)

### the power of scale
![enter image description here](https://miro.medium.com/max/625/1*q-P5aQ7A6VlsfroP3ckg8A.jpeg)
![enter image description here](https://bmk.sh/images/gpt3/perf-small.png)




### scale matters
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
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4NzMzMjY5NTQsLTE5MjE5OTQzNjUsLT
czMjE1MjM5NiwtMTk2Mjc5OTkxNiwzMDc0ODg3NjgsLTkwMjY3
NTQ5OCwtMTc5MDkzNTI2MiwtMjAxNzI0NjIsLTQ1MjY3MjE4Mi
wtMTc0MjIwNzk4NSwxMjAwMzI5MjI5LC05OTY3NzQ2NTYsOTg0
MzQ0NDk3LC0xNTYwNTgxODU4LDExNDIyNzEwMTcsLTMyNzU2NT
MwNSwtMTEyNDk5ODIyMiwyMDYwMjk1MTk1LDQ3MDEwMzYyOSwt
NjYyODU0NTM5XX0=
-->