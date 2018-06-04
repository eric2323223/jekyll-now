# 漫谈Loss function
说起机器学习就不能不提到loss function，因为所有的机器学习问题都可以抽象成loss function的优化过程。Loss function的设计从根本上决定了机器学习任务的成败，本文我们就聊聊loss function

## 原理
简单来说loss function是一种量化模型拟合程度的工具，我们知道机器学习（监督式机器学习）的基本思想设计一个由参数$\theta$决定的模型$f_\theta$，使得输入$x$经过模型$f_\theta(x)$计算后得到接近真实$y$的结果，模型的训练过程是用标签数据（$x_i, y_i$）输入模型$f_w(x_i) = \hat y_i$并计算预测值和真实值的差距$L_w$，求$w$使得$L_w$取最小值，这时模型$f_x$达到最优状态，那么如何判断模型和现实的接近程度呢？如何判断模型已经足够好呢？loss function可以回答这些问题，loss function的loss表示了模型和真实的差距$L_w(\hat y, y)$，当这个距离达到最小值的时候我们就认为模型达到最好的状态。所以机器学习实际上是一个求loss function最小值的问题，radent om/im

## 特性
### Gradient based(GD) optimization
与数学中的求极值问题不同的地方是，机器学习中的求极值使用。机器学习的领域主要使用基于梯度下降（graidient based(GD)的方法，如图一所示对于一个可导的凸函数，从任意一点出发，沿着倒数下降的方向前进直到倒数为零的点，就是函数的最小值。这个方法看上去简单有效，但是在实际的机器学习任务中，模型参数的数量都很大（如VGG16有$1.38*10^8$个参数），这时的loss function的表现会复杂很多，图二展示了模型参数中的两个参数构成的loss function的形态，可见其中有很多区域导数为零，但他们并不是global minimum，算法会在这些区域收敛，这时模型并不具备最优的性能。
![](https://cdn-images-1.medium.com/max/1600/1*t6OiVIMKw3SBjNzj-lp_Fw.png)

![](https://i.stack.imgur.com/TY1L1.png)
#### 那么为什么还要使用GD呢？
数学意义上的的优化问题一般有两类解法，一个是解析方法（analytical optimization），适用于在解析解（closed-form solution），另一种是迭代优化（iterative optimization）方法用于不存在解析解的情况，GD就属于迭代优化的一种典型方法。所有基于神经网络的的优化过程由于存在nolinear activation所以不存在解析解使得$\frac {dl(w)}{dw}=0$，因此只能使用GD方法
- there is no closed form solution
- it is computational impossible to use analytical solution when data is huge
#### what problem caused by GD?
根据GD在导数为0处收敛的特性，可知高维loss function中GD可能会收敛于如下区域
- 0 gradient is not nessisarily global minimum
   - flat region
   - saddle point
   - global minimum is not differenciable

#### how to solve these problem?
- choose better loss function
    - surrogate loss function
- SGD  
- Parameter initialization
### Non-convexity


![](https://www.cs.umd.edu/~tdescent）
的方法求loss function的最小值，这使得loss function的优化问题具有

- convex vs. non-convex
- Regularization
- semantics

## 分类

### By purpose
- classification
- regression
- multi-task learning
### By ...

- MSE
- Cross entropy
- Cosine loss
- Contrastive loss
- CTC
- Triplet loss

## 设计
- loss functin semantic
- strict theoretical minimum of 0
- Convergence
- differenciable

#### Example of loss function design

## 总结



### Distance-based Loss function
### Prediction error-based loss function 



## reference
- [Wiki page](https://en.wikipedia.org/wiki/Loss_function)
- [quora](https://www.quora.com/When-is-square-loss-not-good-for-loss-function-for-regression)
- [How do you decide which loss function to use for machine learning?](https://www.quora.com/How-do-you-decide-which-loss-function-to-use-for-machine-learning)
- [Some thoughts about design of loss functions](https://www.ine.pt/revstat/pdf/rs070102.pdf)
- [On the Design of Loss Functions for Classification: theory, robustness to outliers, and SavageBoost](https://papers.nips.cc/paper/3591-on-the-design-of-loss-functions-for-classification-theory-robustness-to-outliers-and-savageboost.pdf)
- [On Loss Functions for Deep Neural Networks in Classification](https://arxiv.org/pdf/1702.05659.pdf)
- [The Loss Surfaces of Multilayer Networks](https://arxiv.org/pdf/1412.0233.pdf)
- [Loss Functions and Optimization Algorithms. Demystified](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c)
- [Backpropagation — How Neural Networks Learn Complex Behaviors](https://medium.com/autonomous-agents/backpropagation-how-neural-networks-learn-complex-behaviors-9572ac161670)
- [Objective function, cost function, loss function: are they the same thing?](https://stats.stackexchange.com/questions/179026/objective-function-cost-function-loss-function-are-they-the-same-thing)
- [37 Reasons why your Neural Network is not working](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607)
- [Neural networks: which cost function to use?](https://datascience.stackexchange.com/questions/9850/neural-networks-which-cost-function-to-use)
- [What are the impacts of choosing different loss functions in classification to approximate 0-1 loss](https://stats.stackexchange.com/questions/222585/what-are-the-impacts-of-choosing-different-loss-functions-in-classification-to-a)
- [how-to-choose-last-layer-activation-and-loss-function](https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function)
- [Picking Loss Functions - A comparison between MSE, Cross Entropy, and Hinge Loss](http://rohanvarma.me/Loss-Functions/)
- [About loss functions, regularization and joint losses ](http://christopher5106.github.io/deep/learning/2016/09/16/about-loss-functions-multinomial-logistic-logarithm-cross-entropy-square-errors-euclidian-absolute-frobenius-hinge.html)
- [Machine learning non-differentiable loss functions](http://khanhxnguyen.com/machine-learning-non-differentiable-loss-functions/)
- [Loss function semantics](http://hunch.net/?p=269)
- [Escaping from Saddle Points](http://www.offconvex.org/2016/03/22/saddlepoints/)
- [How to Escape Saddle Points Efficiently](http://www.offconvex.org/2017/07/19/saddle-efficiency/)
- [Surrogate loss functions](http://fa.bianp.net/blog/2014/surrogate-loss-functions-in-machine-learning/)
- [A comparison of loss function on deep embedding](https://www.slideshare.net/CenkBircanolu/a-comparison-of-loss-function-on-deep-embedding)
<!--stackedit_data:
eyJoaXN0b3J5IjpbODU2Mjk1NjI5LC0xMzQ4Nzg3NjMsLTM4OT
IxNDEzOCwtMTUxNDc2NTU0MSwtNjQyOTY5MDcwLC03MTg4MzU0
MiwxMjQ3NzcyNjk5LDE1OTQwNDc1NywxODc3NDk5OTUxLC0xNT
kwNTMxNDE2LDYwMjA1OTIwMyw5OTExMjg0LDEwNTc0NzgxMTgs
LTMxNTMwNzM0LC0xMzcwNTQ5NDM1LC0xMjI3Mzg4MzEwLC0xOD
AxNDA3NjY0LDE2MjI5ODY2MDgsMTQ5Mzc0MjgzLC02MzE3NDQ1
OTJdfQ==
-->