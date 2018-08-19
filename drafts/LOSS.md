 # 漫谈误差函数（Loss function）
说起机器学习就不能不提到误差函数（ loss function），因为所有的机器学习问题都可以抽象成loss function的优化过程。Loss function的设计上决定了机器学习任务的成败，本文我们就来聊聊loss functionaientaseD)e thee
![](https://www.cs.umd.edu/~tomg/img/landscapes/noshort.png)

## 概念/原理
是一种量化模型拟合程度的工具，我们知道机器学习（监督式机器学习）的基本思想设计一个由参数$\theta$决定的模型$f_\theta$，使得输入$x$经过模型$f_\theta(x)$计算后得到接近真实$y$的结果，模型的训练过程是用标签数据（$x_i, y_i$）输入模型$f_\theta(x_i) = \hat y_i$并计算预测值和真实值的差距$L_\theta$，求$\theta$使得$L_\theta$取最小值，这时模型$f_\theta$达到最优状态，那么如何判断模型和现实的接近程度呢？如何判断模型已经足够好呢？loss function可以回答这些问题，loss function的loss表示了模型和真实的差距$L_\theta(\hat y, y)$，当这个距离达到最小值的时候我们就认为模型达到最好的状态。所以机器学习实际上是一个求loss function最小值的问题，radent om/im

$$J(\theta) = \frac{1}{m} \sum L(y_i, \hat y)$$

### 指导优化的方向

## 特性

### Gradient based(GD) optimization
与数学中的求极值问题不同的地方是，机器学习中的求极值使用。机器学习的领域主要使用基于梯度下降（graidient based(GD)的方法，如图一所示对于一个可导的凸函数，从任意一点出发，沿着倒数下降的方向前进直到倒数为零的点，就是函数的最小值。
![](https://cdn-images-1.medium.com/max/1600/1*t6OiVIMKw3SBjNzj-lp_Fw.png)
> #### 如何判断凸函数？
> "_If the function is twice differentiable, and the second derivative is always greater than or equal to zero for its entire domain, then the function is convex._"
> how to check convexity?
> -  function lies above all tangents
>$$f(y)=f(x)+\theta f(x)*(y-x)$$
>- second derivative is non-negative


### 可导性
虽然从数学原理上GD要求loss function连续可导，但在实践中loss function可以存在不可导的点，这是因为计算是使用一组（batch）数据的误差均值进行求导，这样使得落在不可导的点上的概率很低，因此可以对hinge loss这样的不连续的函数使用GD来进行优化。事实上即使在某组数据真的发生小概率事件导致求导失效（使用ML工具如tensorflow计算不可导的点上的导数实际上并不会报错，而是返回该点一侧的导数），由于minibatch GD算法使用了大量的分组，绝大多数可求导的分组仍然可以保证GD在整个数据集上有效运行。

### 非凸性
对于所有的凸函数，使用GD都可以找到最小值，但是在实际的机器学习任务中，由于模型参数的数量都很大（如VGG16有$1.38*10^8$个参数），这时的loss function是凸函数的概率非常低，loss function 的表面会复杂很多，图二展示了模型参数中的两个参数构成的loss function的形态，可见其中有很多区域导数为零，但显然他们并不都是最小值，甚至不是局部最小值，~~GD算法会在这些区域收敛，但这时模型并不具备最优的性能。~~
![](http://1L1.png)
![](https://i.stack.imgur.com/TY1L1.png)

根据GD~~在导数为0处收敛~~的特性，可知在高维loss function中，除了global minimum，GD还可能会收敛于如下关键点（critical points）
- 0 gradient is not nessisarily global minimum
   - flat plateu，是指在一个其中所有的点的导数都为0的区域，在三维空间中的flat region就是水平面，即图1中1, 如何避免陷入plateau一直是一个讨论较多的话题，目前常见的方法有：
	   - 适当的参数初始化
	   - 使用替代误差函数（surrogate function）
	   - 调整优化器（如增加动量momentum）
   - local minimum， 如图1中点2
   - 鞍点（saddle point），是一种导数为零但却不是极值的点，如图1中点3处，鞍点是指在该点上一个纬度。。。由于在高维度loss function的所有导数为0的点中，只有在所有维度同时具有相同的凹凸性（即二阶导数都大于0或小于0）的时候loss function才会处于local minimum（说global minimum），任何一个维度的凹凸性不同于其他的维度都会使loss function处于鞍点，考虑到实际的loss function通常会有万或十万（甚至百万）级的维度数量，因此鞍点是非常普遍的。经过试验发现，使用SGD选择具有动量或可变学习率（adaptive learning rate）的优化器（如Adagrad，Momentum）就可以有效的脱离鞍点
![](https://www.researchgate.net/profile/David_Laughlin2/publication/283946342/figure/fig2/AS:297125729587204@1447851702481/Schematic-of-a-saddle-point-illustrating-their-necessity-in-free-energy-critical-point.png)


![](https://ruder.io/content/images/2016/09/saddle_point_evaluation_optimizers.gif)
简单总结，SGD+合适的optimizer(such as momentum) + 合适的参数初始化可以有效找到非凸函数的minima

> #### 那么为什么还要使用GD呢？ 
> 数学意义上的的优化问题一般有两类解法，一个是解析方法（analytical optimization），适用于在解析解（closed-form solution），另一种是迭代优化（iterative
> optimization）方法用于不存在解析解的情况，GD就属于迭代优化的一种典型方法。所有基于神经网络的的优化过程由于存在nolinear
> activation所以不存在解析解使得$\frac {dl(w)}{dw}=0$，因此只能使用GD方法
> - there is no closed form solution
> - it is computational impossible to use analytical solution when data is huge
problem?

### 泛化（generalization）
机器学习的最终目的是提高对未知的输入进行判断的准确率，也就是提高泛化能力。Loss function虽然可以引导GD进行模型的优化，但是一个常见的问题是模型虽然达到了很高的训练准确率，但是泛化能力并没有提高甚至反而降低，这就是过拟合（over fitting）现象，如图所示，蓝色表示一个过拟合的模型，它是一个过度复杂的函数。这种问题源自于模型为了提高训练准确率学习了训练数据中的噪声从而导致模型和真实规律产生偏差。正则化（regularization）就是解决过拟合问题的常见方法之一，它把原误差函数和参数的模（norm）相加形成新的目标函数（objective function），在使用GD对目标函数求最小值。这么做的目的在于降低参数维度从而增加模型的泛化能力，这是由于参数的模变成了目标函数的一部分，因此GD也会尽力降低参数的模，而参数的模和参数的维度正相关，因此GD会降低参数的维度，从而最终实现增强泛化能力的目的。 
![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Regularization.svg/354px-Regularization.svg.png)
- L1 L2 in loss function and regularization
![](https://cdn-images-1.medium.com/max/1600/1*o6H_R3Do1zpch-3MZk_fjQ.png)


## 误差函数的分类
根据不同类型机器学习任务可以将loss function主要可以以下三类：
- 适用于回归问题（Regression）的误差函数，这种误差函数的目标是量化推测值和真实值的逻辑距离，理论上我们可以使用任何距离计算公式作为误差函数。实践中为常用的是以下两种距离：
	 - MSE（Mean Squared Error，L2）
$$MSE=\frac{1}{n}\sum_{i=1}^n (y-\hat y)^2$$
	- MAE（Mean Absolute Error， L1）
$$MAE=\frac{1}{n}\sum_{i=1}^n \mathopen|y-\hat y \mathclose| $$
- 适用于分类问题（classification）的误差函数，分类问题的目标是推测出正确的类型，一般使用概率描述推测结果属于某种类型的可能性，因此误差函数就需要能够计算两个概率分布之间的”距离“，最常用的此类方法是
	- Cross entropy loss （Maximum likelyhood estimation)
$$CE(p,q) = -\sum_x p(x) \log q(x)$$
![](https://datawookie.netlify.com/img/2015/12/log-loss-curve.png)
	- MSE/MAE with thresh hold
- 多任务问题
此类问题是指使用一个模型同时学习多个指标，比如使用深度学习解决计算机视觉中的目标定位（object localization）问题，模型需要同时学习对象类型和对象位置两个指标，因此loss function需要具备同时衡量类型误差和位置误差的能力，常见的做法是先分别设计类型误差函数$L_{class}$和位置误差函数$L_{position}$，在按一定比例合并这两个误差从而形成一个新的误差函数$L$来综合的反应总误差
$$L_{class} = CE$$

$$L_{position} = MSE$$

$$L = \alpha L_{class} + \beta L_{position}$$


## 设计
Loss function不仅仅光只是误差的度量衡量的工具，更重要的是GD会为了不断缩小误差而根据loss function规定的方向（导数方向）调整模型参数，因此可以说loss function它决定了模型学习的目标。使用过不同的loss function我们可以在完全相同的模型架构（model architeture）上学习不同的模型参数，来达到不同的目的。比如，在多层卷积神经网络架构上使用cross entropy loss可以判断图像对象的类型（是猫还是狗），而同样的网络架构配合triplet loss则可以用来提取分辨不同个体的特征（面部识别）。从某种程度上说模型设计决定了了模型的能力，loss function设计决定了模型学习的方向，（划船的比喻）
那么如何设计（或者选择）loss function呢？我们可以从以下几个方面考虑
### 任务目标
决定loss function设计的最重要的因素就是任务目标，有时任务目标和loss function的关系很直接，比如MSE， 有时他们的关系就不那么明显，需要一些的专业知识（domain knowledge）才能和loss function建立联系，比如CTC loss。另外对目标的理解程度也很关键，有时一些细节会对loss function的设计起到关键的作用，比如MSE和MAE是相似的loss function，如何选择取决于任务目标，如果需要避免较大误差则应选择MSE。

无论如何，任务都应该是设计loss function最先考虑的东西。

### 可计算性
通过需求分析得到了初始的误差函数之后，还需要进行数学上的分析来确保它能在GD算法下高效的运行，毕竟整个模型的学习是由大量的误差函数求导过程组成的，误差函数对学习效率的影响是决定性的。
比如
- log likelihood example: why log?
	- log is monotonic
	- much easier to computer joint
	- cross entropy and maximum likelihood estitmation


### 替代误差函数（Surrogate loss function）

有些情况下根据问题目标得到的loss function很难使用GD求极值，例如左图中的loss function在所有可导处导数都是0（水平区域）， 意味着GD无法工作。这时可以使用一个如右图所示的近似的凸函数进行模拟，通过求代理误差函数的最小值来实现优化原来的误差函数的目的。当然这只是一个理想化的例子，并且建立在了解loss function的形态的基础上，但是实际中这种信息通常不容易得到，因此替代损失函数在实际中如何应用是一个比较复杂的问题，有待于进一步研究。

![](http://fa.bianp.net/blog/images/2014/loss_01.png)

![](http://fa.bianp.net/blog/images/2014/loss_log.png)

## 设计实例-人脸识别
从人脸识别探讨loss function设计。
### 需求分析
人脸识别是一个非常有用的功能，它的用户很广，门禁，安保都是典型的使用场景。从机器学习的实现角度来看，应用于门禁的人脸识别有如下特征：
- 单个个体的训练数据较少
- 需要检测未知个体
- 
- 
### 误差函数设计
基于前两条特征，简单CE误差函数显然是不适合的。特别是第二条特征，要求模型能够直接判别未知的个体，这就排除了使用图像识别的方法（每个不同个体都是一个类型）。传统的面部识别技术设计了一系列指标，如双眼的距离，鼻尖到嘴的距离等来作为标识不同个体的特征，我们也可以使用相同的思路来通过类似的特征分离不同的个体，只不过这些特征并不是提前设计好的，而是通过神经网络学习出来的。由此，我们的模型的目标就变成了关键特征提取。
那么如何引导模型选取合适的关键特征呢？最直接的方法就是不做筛选使用全部的特征来计算不同个体特征之间的距离，当不同个体te'zheng这就是contrasitive loss

Large margin softmax loss function 在论文中介绍了一种有趣方法，我们知道
- 面部识别和图像识别的区别
	- 面部识别的目标是识别不同环境中的某一类（某个人）的面部特征
- why naive CR is not working
- 欧式距离
- 角度
	- weight normalization
	- margin
 use regression such as mse in classification (consider margin)
- softmax -> contrastive loss -> triplet loss
				- -> center loss

##Convergence
- differenciable


## 总结
误差函数是机器学习的核心之一，





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
eyJoaXN0b3J5IjpbLTQ0MjEyMDcwNywtMjU5MzUyNjgsLTE4Mz
Y2Mjg1OTcsMTc5MjUxOTUzOSwxMjM4NzY1NjAyLC0yMDQ3Njc3
OTQyLC0xMDAyOTYzOTczLDQ5ODgyNDM4MSwtMTU1MDUyMDg2NS
wtMTU1MDUyMDg2NSwtMjQzMzUxOTY5LDExMjgwNzc3LC0xNjky
Nzc4MDY4LDU5NTc3MTgwMCwtNzIzMTY1OTc4LC0xMTEwNDkwND
Q5LC00MjQxNDAyMjksNTkwMzkyMTYwLDEyMTAxOTgxNzYsMTk4
MDI4NjMxNl19
-->