 # 漫谈误差函数（Loss function）
说起机器学习就不能不提到误差函数（ loss function），因为所有的机器学习问题都可以抽象成loss function的优化过程。Loss function的设计从根本上决定了机器学习任务的成败，本文我们就来聊聊loss function
![](https://www.cs.umd.edu/~tomg/img/landscapes/noshort.png)

## 概念/原理
简单来说loss function是一种量化模型拟合程度的工具，我们知道机器学习的基本思想设计一个由参数$\theta$决定的模型$f_\theta$，使得输入$x$经过模型$f_\theta(x)$计算出的预测值$\hat y$尽量接近真实值$y$，模型的训练过程是用训练数据（$x_i, y_i$）输入模型$f_\theta(x_i) = \hat y_i$并计算预测值$\hat y$和真实值$y$的差距$L(\hat y, y)$，求$\theta$使得$L(\hat y, y)$取最小值，这时模型$f_\theta$达到最优状态，那么如何判断模型和现实的接近程度呢？如何判断模型已经足够好呢？loss function可以回答这些问题，loss function的loss表示了模型和真实的差距$L(\hat y, y)$，当这个距离达到最小值的时候我们就认为模型达到最好的状态。所以机器学习实际上是一个求loss function最小值的问题，radent om/im

$$J(\theta) = \frac{1}{m} \sum L(y_i, \hat y)$$

### 指导优化的方向

## 特性

### Gradient based(GD) optimization
与数学中的求极值问题不同的地方是，机器学习中的求极值使用。机器学习的领域主要使用基于梯度下降（gradient based(GD)的方法，如图一所示对于一个可导的凸函数，从任意一点出发，沿着倒数下降的方向前进直到倒数为零的点，就是函数的最小值。

### 可导性
虽然从数学原理上GD要求loss function连续可导，但在实践中loss function可以存在不可导的点，这是因为计算是使用一组（batch）数据的误差均值进行求导，这样使得落在不可导的点上的概率显著降低，因此可以对hinge loss这样的不连续的函数使用GD来进行优化。事实上即使在某组数据真的发生小概率事件导致求导失败，由于minibatch GD算法使用了大量的分组，绝大多数可求导的分组仍然可以保证GD在整个数据集上有效运行。

> #### 如何判断凸函数？
> "_If the function is twice differentiable, and the second derivative is always greater than or equal to zero for its entire domain, then the function is convex._"
> how to check convexity?
> -  function lies above all tangents
>$$f(y)=f(x)+\theta f(x)*(y-x)$$
>- second derivative is non-negative

![](https://cdn-images-1.medium.com/max/1600/1*t6OiVIMKw3SBjNzj-lp_Fw.png)
### 非凸性
对于所有的凸函数，使用GD都可以找到最小值，但是在实际的机器学习任务中，由于模型参数的数量都很大（如VGG16有$1.38*10^8$个参数），这时的loss function是凸函数的概率非常低，loss function 的表面会复杂很多，图二展示了模型参数中的两个参数构成的loss function的形态，可见其中有很多区域导数为零，但显然他们并不都是最小值，甚至不是局部最小值，~~GD算法会在这些区域收敛，但这时模型并不具备最优的性能。~~
![](https://i.stack.imgur.com/TY1L1.png)

根据GD~~在导数为0处收敛~~的特性，可知在高维loss function中，除了global minimum，GD还可能会收敛于如下区域
- 0 gradient is not nessisarily global minimum
   - flat region，即图1中1
   - local minimum， 如图1中点2
   - 鞍点（saddle point），如图1中点3处，鞍点是指在该点上一个纬度。。。~~事实上，on the surface of a high dimensional loss function, saddle points take majority part of all 0-gradient points, consider a loss function with 100000 parameters, soppose there are  50% possiblity a 0-gradient point is at its minimum and 50% possiblity at its maximum, the possibilty of this point being global/local minimum is $0.5^{100000} \approx 1*10^{-30103}$~~ 
![](https://www.researchgate.net/profile/David_Laughlin2/publication/283946342/figure/fig2/AS:297125729587204@1447851702481/Schematic-of-a-saddle-point-illustrating-their-necessity-in-free-energy-critical-point.png)

> #### 那么为什么还要使用GD呢？ 
> 数学意义上的的优化问题一般有两类解法，一个是解析方法（analytical optimization），适用于存在解析解（closed-form solution）的优化问题，另一种是迭代优化（iterative
> optimization）方法用于不存在解析解的情况，GD就属于迭代优化的一种典型方法。所有基于神经网络的的优化过程由于存在nolinear activation所以不存在解析解使得$\frac {dl(w)}{dw}=0$，因此只能使用GD方法
> - there is no closed form solution
> - it is computational impossible to use analytical solution when data is huge

![](http://ruder.io/content/images/2016/09/saddle_point_evaluation_optimizers.gif)
简单总结，SGD+合适的optimizer(such as momentum) + (random initilization)可以有效找到非凸函数的minima

>#### how to escapte from Plateaus
> It's still a hard problem. Surrogate loss function can help, for example in http://fa.bianp.net/blog/2014/surrogate-loss-functions-in-machine-learning/

### 泛化（Generalization）
机器学习的最终目的是提高对未知的输入进行正确预测的准确率，也就是提高泛化能力。Loss function虽然可以引导GD进行模型的优化，但是一个常见的问题是模型虽然达到了很高的训练准确率，但是泛化能力并没有提高甚至反而降低，这就是过拟合（over fitting）现象。这种问题源自于模型为了提高训练准确率学习了训练数据中的噪声从而导致模型和真实规律产生偏差。正则化（regularization）就是解决过拟合问题的常见方法之一，它的原理是把参数加入loss function作为新的loss function，这样可以避免为了适应训练数据而产生过于复杂模型而。。。
- L1 L2 in loss function and regularization
![](https://cdn-images-1.medium.com/max/1600/1*o6H_R3Do1zpch-3MZk_fjQ.png)


## 常见误差函数
根据不同类型机器学习任务可以将loss function主要可以以下三类：
- 回归问题（Regression），这种误差函数的目标是量化推测值和真实值的逻辑距离，理论上我们可以使用任何距离计算公式作为误差函数。实践中为常用的是以下两种距离：
	 - MSE（Mean Squared Error，L2）
$$MSE=\frac{1}{n}\sum_{i=1}^n (y-\hat y)^2$$
	- MAE（Mean Absolute Error， L1）
$$MAE=\frac{1}{n}\sum_{i=1}^n \mathopen|y-\hat y \mathclose| $$
- 分类问题（classification），分类问题的目标是推测出正确的类型，一般使用概率描述推测结果属于某种类型的可能性，因此误差函数就需要能够计算两个概率分布之间的”距离“，最常用的此类方法是
	- Cross entropy loss （Maximum likelyhood estimation)
$$H(p,q) = -\sum_x p(x) \log q(x)$$
![](https://datawookie.netlify.com/img/2015/12/log-loss-curve.png)
	- MSE/MAE with thresh hold
- 多任务问题（multi task learning)，此类问题是指使用一个模型同时学习多个指标，比如使用深度学习解决计算机视觉中的目标定位（object localization）问题，模型需要同时学习对象类型和对象位置两个指标，因此loss function需要具备同时衡量类型误差和位置误差的能力，常见的做法是先分别设计类型误差函数和位置误差函数，在按一定比例合并这两个误差从而形成一个loss function来综合的反应总误差
$$L_{class} = $$
$$L_{position} = $$
$$L = \alpha L_{class} + \beta L_{position}$$

## 设计

### (loss functin) semantic
- outliers effect
  - MSE vs MAE
  虽然MSE和MAE都能用于regression预测，但是由于MSE对于大误差有更大的惩罚，所以更适合需要避免大误差的预测场景。
  - 0-1 loss is not good for GD because it's gradient is always 0, thus GD cannot learn anything.
- strict theoretical minimum of 0
~~- Convergence~~
### Surrogate loss function
有时根据问题目标得到的loss function很难使用GD进行优化，例如0-1 loss function 在所有可导处导数都是0， 意味着GD无法工作，但是如果
![](fa.bianp.net/blog/static/images/2013/loss_functions.png)

### computation effort
- log likelihood example: why log?
	- log is monotonic
	- much easier to computer joint likelihood
- experiment


## 设计样例
从人脸识别探讨loss function设计。
- 面部识别和图像识别的区别
	- 面部识别的目标是识别不同环境中的某一类（某个人）的面部特征
- 
 use regression such as mse in classification (consider margin)
- softmax -> contrastive loss -> triplet loss
				- -> center loss

#### Example of loss function design

## 总结





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
- [神经网络如何设计自己的loss function，如果需要修改或设计自己的loss，需要遵循什么规则](https://www.zhihu.com/question/59797824)
- [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTc4MTU2NTIzOCwtMTU4NTI3NjE4MywtMT
I5ODkyOTEwMCwtMTk1MTYzNzIyOSwtODIzNTI2MDQ5LDEwMjU2
NTUzNywtMzc4MDYwNTgxLC0xMTExMTc2NzY1XX0=
-->