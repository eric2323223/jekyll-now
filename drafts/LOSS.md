# 漫谈Loss function
说起机器学习就不能不提到loss function，因为所有的机器学习问题都可以抽象成loss function的优化过程。Loss function的设计从根本上决定了机器学习任务的成败，本文我们就聊聊loss function

## 原理
简单来说loss function是一种量化模型拟合程度的工具，我们知道机器学习（监督式机器学习）的基本思想设计一个由参数$\theta$决定的模型$f_\theta$，使得输入$x$经过模型$f_\theta(x)$计算后得到接近真实$y$的结果，模型的训练过程是用标签数据（$x_i, y_i$）输入模型$f_w(x_i) = \hat y_i$并计算预测值和真实值的差距$L_w$，求$w$使得$L_w$取最小值，这时模型$f_x$达到最优状态，那么如何判断模型和现实的接近程度呢？如何判断模型已经足够好呢？loss function可以回答这些问题，loss function的loss表示了模型和真实的差距$L_w(\hat y, y)$，当这个距离达到最小值的时候我们就认为模型达到最好的状态。所以机器学习实际上是一个求loss function最小值的问题，

## 特性
### Gradient based optimization
与数学中的求极值问题不同的地方是，机器学习中的求极值使用graident based的方法的的在学习的开始阶段我们最常见到的loss function是这样的，如图一所示

why GD?
- there is no closed form solution
- it is computational impossible to use analytical solution when data is huge
what problem caused by GD?
- 0 gradient is not nessisarily global minimum
   - flat region
   - saddle point
### Non-convexity
### 不唯一性
- surrogate loss function
### 

how to solve these problem?
- choose better loss function
    - surrogate loss function
- SGD   
how to escape from saddle point
what is global minimum is not differenciable?
how can SGD help?


1.  convex
当loss function是convex函数时，最小值存在于倒数$\frac {\delta L}{\delta w}$为零时， 图1表示一维$w$的loss function， 图2表示二维$w = (\beta_0, \beta_1)$的loss function
- convex loss function
![](https://cdn-images-1.medium.com/max/1600/1*t6OiVIMKw3SBjNzj-lp_Fw.png)


-2. non-convex function

![](https://i.stack.imgur.com/TY1L1.png)
![](https://www.cs.umd.edu/~tomg/img/landscapes/noshort.png)
2. non-convex
对于
4. 	
	- local minima
		- SGD
		- initialization
		- 
	- saddle point



minimized the distance between expected value and ground truth value

- convex vs. non-convex
- Regularization
- semantics

## 分类

## 设计

## 总结

- 0/1 loss function
- surrogate-loss-functions

### Distance-based Loss function
### Prediction error-based loss function

## Purpose
### Classification
- Cross entropy
### Regression
 MSE(Mean Square Error)
 Triplet
 C
# Contrastive
 Square loss


## Design of loss function
### Multi-task learning
### Auxiliary loss?
some test will be required to ensure it work like expected


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
eyJoaXN0b3J5IjpbOTE3OTk4ODMzLC0yNDU2OTE4MDUsMTQxMz
AxMjEwNSwtMTM5NTY1OTYzOSwtMTgyNDI5NTMzMiwtMTU5Njgw
NTAsMTEwOTU4OTc4NiwtNjQ1NzI1ODg4LDI1MTkxNDk3NCwtOD
IyMTY0MTg1XX0=
-->