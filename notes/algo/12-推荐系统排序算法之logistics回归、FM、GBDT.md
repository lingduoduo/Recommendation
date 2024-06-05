12：推荐系统排序算法之logistics回归、FM、GBDT

我们在[上一篇文章](http://mp.weixin.qq.com/s?__biz=Mzk0MzE3MDEyNQ==&mid=2247494785&idx=1&sn=bcf5027f32933e8ff4ba28b054c17527&chksm=c3355ba3f442d2b58032174f3fd3625f59d91a11c120fbbe964b90ad0b68b7c887920b3d65e3&scene=21#wechat_redirect)中介绍了5种最基础的、基于规则策略的排序算法，那些算法是在没有足够的用户行为数据的情况下不得已才采用的方法，一旦我们有了足够多的行为数据，那么我们就可以采用更加客观、科学的机器学习排序算法了。

本章我们就来讲解3个最常用、最基础的基于机器学习的排序算法，分别是logistics回归、FM（分解机）和GBDT（**G**radient **B**oosting **D**ecision **T**ree）。这些算法原理简单、易于工程实现，并且曾经在推荐系统、广告、搜索等业务系统的排序中得到了大规模采用，是经过实践验证的、有业务价值的方法。

虽然随着深度学习等更现代化的排序算法的出现，这些比较古老的算法没有像之前那么被大家津津乐道了，但是他们在某些场景下还是会被采用的，在当前（甚至未来）不会退出历史舞台。熟悉这些算法对大家更好地理解排序的原理及对后面更复杂的排序算法的理解是大有裨益的，其实它们的一些思路对启发更高阶的算法是非常有帮助的，甚至它们就是某些更高阶算法的组成部分。

下面我们就开始分别介绍这3类算法。在介绍算法原理的同时，我们会简单提一下它们的推广与拓展及这些推广与拓展在大厂的应用，不过我们不会深度讲解，我会给出相关的论文，感兴趣的读者可以查看论文进一步学习。

**12.1 logistics回归排序算法**

logistics回归模型是最简单的线性模型，原理简单、工程实现容易，因此是最基础的排序模型。下面我们从算法原理、特点、工程实现、业界应用等4个方面来展开说明。

 **12.1.1 logistics回归的算法原理**

logistic回归模型(**LR**)(见下面公式1，其中 ![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3icElbbaNF7KQNs3iaRdWghND0V8CSyJcVicXKib8F5ibdVxayWdkCk41Y7w/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1) 是特征，![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3UOmC018Bicy3Bjbic2OQUVDnicLGComfRXPDIZs9M0G4jqFSbicTKNyjqQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1) 是模型的参数)是简单的线性模型，原理简单，易于理解，并且非常容易训练，对于一般的分类及预测问题，可以提供简单的解决方案。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx39BxgzRON4vRibeoLIw8h0BOaDII0IPUictUJLicOVQHzK6TcqIJVgKf6g/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

公式1：logistics回归模型

为什么说logistics回归是线性模型呢？其实logistics回归是将线性模型做了一个logistics变化获得的。下面的logistics函数![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3aiaG2fQIGChFx1GJAjkvoUicBWgibpUqH8NBOuojGbfKUDicic0ZaFFvhEw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)就是logistics变换，对比公式1和公式2，大家应该可以看到logistics回归模型就是将线性模型通过logistics变换获得的。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3puyRQWWC5bL9zHmzkCSCDnT80I7kITQsOzyiaTGurib4Uicmp8gwUFxLw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

公式2：logistics变换（函数）

logistics函数是一个S曲线，得到的结果在0和1之间，x的值越大，s(x)越接近1，x的值越小，s(x)越接近0，具体曲线如下面图1。

 

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3VtnpaDzg9r9QhFDAE19dy2he6ia7hfGUNichOibbpXD6BJjrHM5K3A4GQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图1：logistics函数的图像

 

通过将线性函数做logistics变换的最大价值是将预测结果![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3e9ibVEaUEnoJSIDtmqzZ5nfzvBEDDicgiazIV5FNJpHACEF6o7kic2ZVZw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)（公式1的左边）变换到0到1之间，因此logistics回归可以看成是预测变量的概率估计，所以对于预测点击率（或二分类问题）这类业务是非常直观实用的。

对于推荐系统来说，上面的特征![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3icElbbaNF7KQNs3iaRdWghND0V8CSyJcVicXKib8F5ibdVxayWdkCk41Y7w/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)就是各类特征（用户维度特征、物品维度特征、用户行为特征、上下文特征等），而预测值 ![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3MyVnS3XkGWXIsKPEEcooAWfZFibDrAgOfDe7ib64mI6ciahDice9ExxQFA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1) 就是该用户对物品的点击概率。那么最终的推荐排序就是先计算该用户对每个物品（这些物品是召回阶段的物品）的点击概率，然后基于点击概率降序排列，再取topN作为最终的推荐结果。下面图2就可以很好地说明logistics回归模型怎么用于推荐系统相关的预测中。

$\hat{y}(x)=\frac{1}{1+\exp \left(w_0+\sum_{i=1}^n w_i x_i\right)}$

图2：logistics回归模型用于推荐系统排序

 **12.1.2 logistics回归的特点**

从上面的介绍我们可以知道，logistics模型非常简单，基本有一点数学知识的人都能理解，这个模型也非常容易用到推荐系统排序中。但logistics回归模型的弱点也非常明显：怎么构建交叉特征这个任务是logistics回归不能帮助我们的（构建交叉特征的过程相当于对模型进行了非线性化，可以提升模型的预测能力）。

logistics回归模型的特征之间是彼此独立的，无法拟合特征之间的非线性关系，而现实生活中往往特征之间不是独立的而是存在一定的内在联系。以新闻推荐为例，一般男性用户看军事新闻多，而女性用户喜欢娱乐八卦新闻，那么可以看出性别与新闻的类别有一定的关联性，如果能找出这类相关的特征，是非常有意义的，可以显著提升模型预测的准确度。

当然，我们也可以利用人工去做特征的交叉获得特征之间的非线性关系，不过这需要建模人员对业务非常熟悉，知道哪些特征之间交叉对预测目标是有帮助的，有时还免不了进行大量的尝试。这可能也是logistics回归模型的缺点。实际上，LR模型最大的缺陷就是人工特征工程，耗时费力，浪费大量人力资源来筛选、组合非线性特征。

LR模型是CTR预估领域早期最成功的模型，也大量用于推荐算法排序阶段，大多工业推荐排序系统通过整合人工非线性特征，最终采用这种“线性模型+人工特征组合引入非线性”的模式来训练LR模型。因为LR模型具有简单、方便易用、解释强、易于分布式实现等诸多好处，所以目前工业上仍然有不少业务系统采取这种算法。

 **12.1.3 logistics回归的工程实现**

logistics回归模型是简单的线性模型，利用梯度下降算法（如SGD）就可以简单训练。像scikit-learn就包含logistics回归模型（参考类：sklearn.linear_model.LogisticRegression）。如果数据量大，也可以利用Spark MLlib中的logistics回归（见参考文献1）实现，这里不赘述。

 **12.1.4 logistics回归在业界的应用**

我们要讲的第一个应用是关于logistics回归模型怎么用于实时场景中。在这个方向上，Google在2013年提出了**FTRL**(**F****ollow-****T****he-****R****egularized-****L****eader**)算法用于广告点击率预估，该方法可以高效地在线训练LR模型，在Google及国内公司得到了大规模的采用，广泛用于广告点击率预估和推荐排序中。想了解的读者可以阅读参考文献2。 

我们下面说另外一个在阿里的应用。参考文献3中，阿里提出了一种分片线性模型，其核心思想是分而治之。首先对样本分为m类，在每类中应用logistics回归，由于不同类样本的特性不一样，所以logistics回归的参数也是不一样的。不同类之间采用softmax函数作为权重（参见下面公式中![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3QMnFguszm8VX8qecoLpE0xplknAFgRKTbYw8yEFlibREU7MbjF9IJfA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)部分），见下面的公式3。当m=1时就是普通的logistics回归模型，当m大时，预估的会越准，但是这时参数也越多，需要更多的训练样本、更长的训练时间才能训练出有效果保证的模型。

 

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx36mEfe86XRMib0AicUiaQon8OGXNgaGqXUUM0YxmOUfib088RLOyfqibXs4A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

公式3：分片线性模型

 

如果从更现代的视角来看，上面的系数![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3nicbMFgSELIlGDo4TJicOPhicwuuKUINxfhBQ0iauibOWSTe4fNjyb2gmhQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)类似于注意力机制的注意力参数。关于这个模型的细节介绍，大家可以阅读参考文献3，这里不展开说明了。

 

**12.2 FM排序算法**

分解机最早由Steffen Rendle于2010年在ICDM会议(**I**ndustrial **C**onference on **D**ata **M**ining)上提出，它是一种通用的预测方法，即使在数据非常稀疏的情况下，依然能估计出可靠的参数，并能够进行比较精准的预测。

与传统的简单线性模型不同的是，因子分解机考虑了特征间的交叉，对所有特征变量交互进行建模（类似于SVM中的核函数），因此在推荐系统和计算广告领域关注的点击率CTR（**C**lick-**T**hrough **R**ate）和转化率CVR（**C**on**V**ersion **R**ate）两项指标上有着良好的表现。此外，FM模型还具备可以用线性时间来计算，可以整合多种信息，以及能够与许多其他模型相融合等优点。下面我们从算法原理、参数估计、计算复杂度、模型求解、排序方法等5个维度展开介绍。

 **12.2.1 FM的算法原理**

我们在12.1节中讲到了logistics回归不具备自动组合特征的能力，这是它的缺点。那么能否将特征组合的能力体现在模型层面呢？也即，是否有一种模型可以自动化地组合筛选交叉特征呢？答案是肯定的。

其实想做到这一点并不难，如图1，在线性模型的计算公式里加入二阶特征组合即可，任意两个特征进行两两组合，可以将这些组合出的特征看作一个新特征，加入线性模型中。而组合特征的权重和一阶特征权重一样，在训练阶段学习获得。我们可以在线性模型中整合二阶交叉特征，得到如下的模型。 

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3aCCoK1Bl14aKn7BT9HxJbFZ7zHrsmPIp4eGaBueMpa1B19wnHM2Tqg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

公式4：二阶线性模型 

上述模型中，任何两个特征之间两两交叉，其中， n代表样本的特征数量，![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3cibWicJSUGtksxacicOpfOvHOvOvZqRpweLmgHLyibj5DT9o7DILIWibjpA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是第i个特征的值，![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx35cFUASCDua6uQiaKdadOdzgia5He0m1bAjzvQjA4IbZZzicDXEREbpb8w/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)、![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3uicuH9t9OYBUJGNKk2YwqibXCBIMPU3iafYibE1Zt6WUhbABvHu2tROytA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1) 、![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3IS7RsnBDhsTvTAcFOBg94wolgZJYcdrb29qeKHm8UUhpmTZm1snQdw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是模型参数，只有当![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3cibWicJSUGtksxacicOpfOvHOvOvZqRpweLmgHLyibj5DT9o7DILIWibjpA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)与![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3Rc0QzAEeeLWnFYaDGiajAlwIKpCHh1HEexbRmfKWTXmhut5iaSPiadhaw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)都不为0时，交叉项才有意义。

虽然这个模型看上去貌似解决了二阶特征组合问题，但是它有个潜在的缺陷：它对组合特征建模，泛化能力比较弱，尤其是在大规模稀疏特征存在的场景下，这个毛病尤其严重。在数据稀疏的情况下，满足交叉项不为0的样本将非常少(非常少的主要原因有，有些特征本来就是稀疏的，很多样本在该特征上是无值的，有些是由于收集该特征成本过大或者由于监管、隐私等原因无法收集到)，当训练样本不足时，很容易导致参数![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3IS7RsnBDhsTvTAcFOBg94wolgZJYcdrb29qeKHm8UUhpmTZm1snQdw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)训练不充分而不准确，最终影响模型的效果。特别是对于推荐、广告等这类数据非常稀疏的业务场景来说(这些场景的最大特点就是特征非常稀疏，推荐是由于标的物是海量的，每个用户只对很少的标的物有操作，因此很稀疏，广告是由于很少有用户去点击广告，点击率很低，导致收集的数据量很少，因此也很稀疏)，很多特征之间交叉是没有(或者没有足够多)训练数据支撑的，因此无法很好地学习出对应的模型参数。因此上述整合二阶两两交叉特征的模型并未在工业界得到广泛采用。

那么我们有办法解决该问题吗？其实是有的，我们可以借助矩阵分解的思路，对二阶交叉特征的系数进行调整，让系数不在是独立无关的，从而减少模型独立系数的数量，解决由于数据稀疏导致无法训练出参数的问题，具体是将上面的模型修改为

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx36ZbgavJGXny05aLDDsBldp4GFZKsvbDm8Sxf6DeIBHT7BvuvXIMHaw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

公式5：FM模型

其中我们需要估计的模型参数是![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx35cFUASCDua6uQiaKdadOdzgia5He0m1bAjzvQjA4IbZZzicDXEREbpb8w/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)、![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3ukTOibdQnic9ProvMf6yqkRNdy93En5vKpgkhqibKHg6CiapGolfJzS0uQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)、![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3YdKnvzOAsgxn24BxkjRflAxsicY6aW7yR0IwZA4sAQJtDZ05xEc0Y4A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)。

其中![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3OMmuECIbf0MJlrLguhQIZibTLw6CbVzt8VMcplSvwzW7GekqibVLSFDQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，是n维向量。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3CwecjYHPkic1Jn1crFwd5G3CgRCVkIQXLvDBj91J6QEjH9QVTaK9MGA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)、![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx32ZF9vRpn3eVeghn5yibYbkeRwYauILYaGLlNwlFykEymrjW56icDbNmw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是低维向量(k维)，类似矩阵分解中的用户或者标的物特征向量表示。V是由![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3CwecjYHPkic1Jn1crFwd5G3CgRCVkIQXLvDBj91J6QEjH9QVTaK9MGA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)组成的矩阵。< , > 是两个k维向量的内积：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3euVEfdpucqt69ibRyXo7BpWTKY9GHcLDzSicLDIJGgLj8ic2jDdQzvWRA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1) 

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3CwecjYHPkic1Jn1crFwd5G3CgRCVkIQXLvDBj91J6QEjH9QVTaK9MGA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)就是我们的FM模型核心的分解向量，k是超参数，一般取值较小(100左右)。

利用线性代数的知识，我们知道对于任意对称的正半定矩阵W，只要k足够大，一定存在矩阵V使得![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx35ShLYg2UX6QV7SsRoCUJ3dgXTQ1FdEtdeNfiaqM292c1hHQ2w7iaC0Aw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)(**Cholesky decomposition**)。这说明，FM这种通过分解的方式基本可以拟合任意的二阶交叉特征，只要分解的维度k足够大(首先，![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3QrJGPrkFfic3FeEYstQHhZ3WsngtG5MvX79fAicJgVZI1vG26qaZMOOA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)的每个元素都是两个向量的内积，所以一定是对称的，另外，分解机的公式中不包含![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3cibWicJSUGtksxacicOpfOvHOvOvZqRpweLmgHLyibj5DT9o7DILIWibjpA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)与![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3cibWicJSUGtksxacicOpfOvHOvOvZqRpweLmgHLyibj5DT9o7DILIWibjpA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)自身的交叉，这对应矩阵![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3QrJGPrkFfic3FeEYstQHhZ3WsngtG5MvX79fAicJgVZI1vG26qaZMOOA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)的对角元素，所以我们可以任意选择![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3QrJGPrkFfic3FeEYstQHhZ3WsngtG5MvX79fAicJgVZI1vG26qaZMOOA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)对角元素足够大，保证![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3QrJGPrkFfic3FeEYstQHhZ3WsngtG5MvX79fAicJgVZI1vG26qaZMOOA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是半正定的)。由于在稀疏情况下，没有足够的训练数据来支撑模型训练，一般选择较小的k，虽然模型表达空间变小了，但是在稀疏情况下可以达到较好的效果，并且有很好的拓展性。

 **12.2.2 FM的参数估计**

对于稀疏数据场景，一般没有足够的数据来直接估计变量之间的交互，但是分解机可以很好地解决这个问题。通过将交叉特征系数做分解，让不同的交叉项之间不再独立，因此一个交叉项的数据可以辅助来估计(训练)另一个交叉项(只要这两个交叉项有一个变量是相同的，比如![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3E8kXX1W1oorRpIoAlcCDn7uRhqibPjLBibgBnHV7IwwdVSnCyzV6uGjg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)与![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3jibYbOrM2uRgqkk3Aib6teiaZf71bibia9lRIibBOJYDdia4YiaZ3c5lVMcPKg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，它们的系数![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3bo2Jiaiajd15wh7KMuruYoSvtPWHg5nc74IMMEszSQcI7kyXPPndzDxw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)和![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3x7hNSL62TWj4nDMP4pb73w5z1r1GwXDFCf86zGYB6ic6C1Ldd2mRibXg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)共用一个相同的向量![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3CwecjYHPkic1Jn1crFwd5G3CgRCVkIQXLvDBj91J6QEjH9QVTaK9MGA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1))。

分解机模型通过将二阶交叉特征系数做分解，让二阶交叉项的系数不再独立，因此系数数量是远远小于直接在线性模型中整合二阶交叉特征的。分解机的系数个数为![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3DTt2jau3NsG6MfuHKSfPm0G2JK6gQrV5APYeumtkpFia6xMj6RO7IaQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，而整合两两二阶交叉的线性模型的系数个数为![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3Yb7h4cI9yktMwDP8xc3nXic835nPiaG5OUibXDvnAu1hr0TRIXUiaIdUpA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)。分解机的系数个数是n的线性函数，而整合交叉项的线性模型系数个数是n的指数函数，当n非常大时，训练分解机模型在存储空间及迭代速度上是非常有优势的。

**12.2.3 FM的计算复杂度**

直接从公式5来看，因为我们需要处理所有特征交叉，所以计算复杂度是![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3gfF8GjNNfgzvpXRKBictOpMChX313yibsFX6B8m79o9emXqqYW4Wf2jQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)。但是我们可以通过适当的公式变换与数学计算，将模型复杂度降低到![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3EFyaNdCvde68Rl5oHDezsA1fvcsUHkNwWA9McNMCgdgPzeFp3Xk5Lw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，变成线性复杂度的预测模型，具体推导过程如下：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3PM25ficeDm3xVFT5kWgd3DBVYJdqIjFu3jn2fxicwAaAXKia5OkUVg1VA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

公式6：FM交叉项的计算可以简化为线性复杂度的计算

从上面公式最后一步可以看到，括号里面复杂度是![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3WcMicKzliakXvjDjLHOH3X8vGsmrQk9X5f3wlPq7VdwqicauG42pdjc6g/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，加上外层的 ![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx37gVH86FGS3wOic4IXkN3690ROUiaCicO5c8ibqxQibSlAuMcemXZ72Bib6Cw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，最终的时间复杂度是![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3EFyaNdCvde68Rl5oHDezsA1fvcsUHkNwWA9McNMCgdgPzeFp3Xk5Lw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)。进一步地，在数据稀疏情况下，大多数特征x为0，我们只需要对非零的x求和，因此，时间复杂度其实是![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3Q4QKkicPs0hOmTSOUZB6MGI7otdMWwLnb1Ebxwqu5oxdv6BQkmXNpxQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx337OIKBwkmkuSgd9urTsHqibMWHSMOB4Q83aq7RJ4eMIoLuJYFiamtdIQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是训练样本中平均非零的特征个数。

由于分解机模型可以在线性时间下计算出结果，对于我们做预测是非常有价值的，特别是对有海量用户的互联网产品，具有极大的应用价值。拿推荐系统来说，我们每天需要为每个用户计算推荐(这是离线推荐，实时推荐计算量会更大)，线性时间复杂度可以让整个计算过程更加高效，可以在更短的时间完成计算，节省服务器资源。

 **12.2.4 FM求解**

分解机模型公式相对简单，完全可导，我们可以用平方损失函数或者logit损失函数来学习FM模型。从12.2.3节的介绍中我们知道分解机模型的值可以在线性时间复杂度计算出来，因此FM的模型参数(![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx32ur7bLhratTbgbap5RXSWCgg1ibjITiaRydL9EljibV3rxx7n41vPmC3w/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1))可以在工程实现上高效地利用梯度下降算法(SGD、ALS等)来训练(即我们可以线性时间复杂度求出下面的![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx38rucvvwdHMWoDnJBV5bYXXW0aJSzqGgXicicGZqEgG7jahibwAFfhBiamQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，所以在迭代更新参数时是非常高效的，见下面的迭代更新参数的公式)。结合公式5和公式6，我们很容易计算出FM模型的梯度如下：

 

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3EEqibSSwxPPKficib0qwUuzjHiaL01IkoSzEdYzc9ue8rp9RlbLxmgoOrw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

 

我们记![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3icGEhL5MbjDmOCN5bc8Hw0p4tUym4NDHlU3QsO0u0cYAibQRwZkVko3g/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)，针对平方损失函数，具体的参数更新公式如下(未增加正则项，其他损失函数的迭代更新公式类似，也可以很容易推导出)：

 

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3Usr1t99mJ8ibQBOFlrxnyjNfb2OWj6bib0GhA7rMvwq0jH4fuyY6FAqg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx36iaeAEEW8tkJ1jA4NtgbvVRK9oE5E3P3exIevJcRcD2Rjd7NlIbCzRw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3JB4xib05FZygRibP7GNjW9xryQUtwpib8QcibS8QCx2qPYXicNFXkaknl4A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

其中，![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3F05oPt5U4dXnebOpA3UWw9ZRVXnNFnYIjhjU23honv8ic58iaXcPPDRQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1) 与i无关，因此可以事先计算出来(在做预测求![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3BRu7Lxq5db6Iu35PHVf9fTbfYUzmicXM03iarIb3BrLguXIrnibKwjPqg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)或者在更新参数时，这两步都需要计算该量)。上面的梯度计算可以在常数时间复杂度![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3vW3xRDEtTy7YJy6pHItBH8n5b2GcGysEdsvJhSQSta2nVoXjF5zCug/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)下计算出来。在模型训练更新时，在![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3EFyaNdCvde68Rl5oHDezsA1fvcsUHkNwWA9McNMCgdgPzeFp3Xk5Lw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)时间复杂度下完成对样本![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx34JBf5pMtaWUsOaSsx80YC4OscBibeLCVFnAd9cA5J26qz166MuvZfcg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)的更新(如果是稀疏情况下，更新时间复杂度是![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3KM86bP22ozQf6rxvdhJkvBquic3aK1Y00odG1WvsNnnqSyAFeLqduOw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1), ![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3pylsW1RfF1uz74nwluqZp4JTaG66ASLbQsJDRX8p3a0Xiae6lcVQjXA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是特征![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3pu7p4ibczp1LROtbczpl1DV5qkhVgzl385MfnLX8LBNX7RoQ6Q2zIbA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)非零元素个数)。

 **12.2.5 FM进行排序的方法**

分解机是一类简单高效的预测模型，可以用于各类预测任务中，主要包括如下2类：

- **回归问题(Regression)**

如果推荐排序预测的是具体的物品的得分，那么![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3BRu7Lxq5db6Iu35PHVf9fTbfYUzmicXM03iarIb3BrLguXIrnibKwjPqg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)直接作为预测项，可以转化为求最小值的最优化问题，具体如下： 

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3SknUNic5FwVMQkEumCSl8U7uG8CSjhw2sdOX2EquD5mS5yibN2vibavOA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

其中D是训练数据集，y是![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3pUS6G5cX3LdclHclyXCQfjEVw6jsOK0QBzux75bA5Qza9pCwlichMxw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)对应的真实值。

- **二元分类问题 (Binary Classification)**

如果推荐排序预测的是二分类问题（比如预测的是用户是否点击），我们可以通过logit loss来训练二元分类问题（即类似logistics回归一样，加上一个logistics变换）。

 上面说完了FM用于回归和分类的两种排序方式，关于FM更详细的介绍读者可以阅读参考文献4。下面来简单介绍2个关于FM的代码框架，方便大家使用。关于FM的开源实现是非常多的。FM的作者之前开源过一个实现方案，读者可以查看参考文献5。如果数据量大，我们还可以利用Spark MLlib，Spark MLlib中有FM的分布式实现，并且可以用于做分类和回归，大家可以查看参考文献6、7，里面有完整的代码案例，这里我们不赘述。如果大家用PyTorch或者TensorFlow，利用它们提供的最优化工具，按照12.2.4的介绍，自己也是非常容易实现的。

 **12.3 GBDT排序算法**

GBDT(Gradient Boosting Decision Tree)是一种基于迭代思路构造的决策树算法，该算法在实际问题中将生成多棵决策树，并将所有树的结果进行汇总来得到最终结果，该算法将决策树与集成思想进行了有效的结合，通过将弱学习器提升为强学习器的集成方法来提高预测精度。GBDT是一类泛化能力较强的学习算法（读者可以查看参考文献8更详细了解GBDT），下面我们从算法原理、推排序应用2个维度来说明。

 **12.3.1 GBDT的算法原理**

GBDT的模型形式如下面公式，其中 ![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx31T8OXUw7XSKqFRsiaDiaV6gmUQMZcIpOWt8EcrZZ98Dh6NHfS6Qfx9kA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1) 是第i棵决策树，其中![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3kaMrpM2bBaRDSJbh9MES51bibGlANfkbsIVDKcKUZgagyjetwCw7eug/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1) 是模型的特征，![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx38sQfRZqiaia4yzW62KpvNqqiceL8QYG6Jr5mfX0ibTZ65IkdB980xuo3Uw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1) 是树的参数，![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3uibyfSOXpMrB23t8h67ehn28QBZSibYGUXDtm8xo9a7txtyrJRvdb8tQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1) 是各棵树的权重。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3ZFyToibwY6THJgvibw9HaHjeicRpenKxWiaqG2xSYMricyhZR1ia1icQiagarg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

GBDT是每次选择一棵树，对现有模型逐步做加法扩展，从而得到最终的强学习器的，也就是通过M次迭代，才学习到最终的模型。从上一次迭代到下一次迭代是学习的模型的残差，下面来简单说明。

 假设我们使用的是平方损失函数，那么从k-1次到k次迭代，我们的损失函数可以表示为如下形式：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3l4YZO5lHhicGgPA4UcoONlrTc4JOiaV9JK0C35HWWGpsRJhn7VJHwEBw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

上式中， ![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3DAHB8MQnsBLtMUt4ICdicQxwmW51Kqspic2FEEMaG0kGeV300JRo5odA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)是当前模型![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx39NEIp2wyia6QG820MbJtK89XC5PTeoBibsZ6dkI3DNEKzSH8Wu0E3YYQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)在样本点![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3O1PUaxSOlaZXVGs3DDnp8x5KoqB27lFm8yMpYr8jaOTVK9GdXibabBA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)第k步的残差，我们学习一个新的树![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3uhzwPQOFIYMk14nxfTlK5siaDovRRnXwszz086yN0ORSyHsvkWrnkPA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)来拟合残差 ![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3DAHB8MQnsBLtMUt4ICdicQxwmW51Kqspic2FEEMaG0kGeV300JRo5odA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1) 。所有样本点上的残差之和就是整个训练样本在第k次迭代的残差（见下面公式）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3VzLXAF5BTuW3Y3ibbsLeIZskLibtzNJr6ictv8W62lnlq08szBYWay26A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

这种多次迭代的过程有点类似高数中求极限，是一个逐步逼近的过程，随着迭代的次数越来越多，残差会越来越小，模型的预测精准度也会越来越高。整个迭代的过程我们可以用如下图示非常清晰的说明。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3hVjdCyPIUwictXiagC44icwg6Zko0KGhGZG0WnLdTqPPVdO99PgRjREjw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图3：GBDT逐步逼近残差的过程

关于GBDT的算法原理我们就介绍到这里了，算法的详细推导过程，读者可以查看参考文献9、10，特别是9中讲解的非常清楚，我们这里不赘述。

 **12.3.2 GBDT用于推荐排序**

上面我们在12.3.1节中介绍了GBDT算法的原理，我们是按照回归任务来介绍的，其实GBDT也是可以用于分类的，参考文献11里面介绍的非常清楚，读者可以自行学习。所以对于推荐排序来说，GBDT是可以用于预测用户对物品的评分及预测用户对物品的点击概率的（二分类问题）。 

目前GBDT的开源实现非常多，比较出名的有XgBoost（参考文献12）、LightGBM（参考文献13）。如果数据量大，也可以采用开源的分布式实现，目前Spark MLlib中是有GBDT的实现的，读者可以从参考文献14、15中了解具体情况，里面有比较完整的demo代码式例。其实，XgBoost和LightGBM都是支持Spark的，读者可以从参考文献16、17中了解细节。 

DBDT是一种集成模型，具备集成模型的优点，它的泛化能力很好，预测精准度高，并且离散特征、数值特征都可以使用，不同特征即使量级不一样也不会影响模型的效果，因此是一种非常值得尝试的排序模型。

最后我们介绍一下Facebook在2014年发表的一篇论文（参考文献18），这篇文章非常创新地将GBDT和前面介绍的logistics回归模型结合起来了。先对样本利用GBDT来训练一遍，模型的叶子节点当做特征，然后将特征灌入logistics模型再次训练，这两个过程是解耦的，下面图4说明了这个算法的过程。这么做的价值体现在：先用GBDT构建特征，这些特征通过GBDT模型的训练是非线性的特征，肯定包含了各种原始特征的交叉了，这就解决了logistics回归需要人工构建特征并且特征不方便交叉这两个问题，可谓一举两得。这种方法也包含了现在深度学习推荐算法的影子，深度学习一般都是通过别的嵌入方法获得特征，然后灌入深度学习模型，利用MLP来训练模型。这里GBDT起嵌入的作用，而logistics类似MLP神经网络（其实logistics可以看成最简单的神经网络，它只有输入节点和输出节点，没有隐藏层）。 

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aD40Sib0kia78K9H9axM6ITsfXVXNaXqx3iceTD57LWp5hlq9nZu4biafxkR2WL8SK6CEbqh6gSziaXXfgLfP4FgzZA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

图4：GBDT+LR的模型结构

 **总结**

本章我们讲解了3类最常用、最基础的推荐排序模型，分别是logistics回归、FM和GBDT，它们都能用于推荐排序的回归和分类问题，它们曾经也是各个大厂主流的推荐、搜索、广告排序算法。

 这3个算法虽然原理简单易懂，工程实现也不复杂（有很多开源的工具供大家使用），但它们包含的思想是值得大家学习的，它们目前也是各种深度学习算法的构件（比如我们在下一章中讲到的wide & deep中就用到了logistics组件，DeepFM中就用到了FM组件），并且在某些场景下（比如数据量不太多、计算资源不足）也是不二之选。本章的讲解就到这里了，下一章我们会讲解深度学习等高阶推荐排序算法。

 **参考文献**

1. https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression

2. Ad Click Prediction- a View from the Trenches

3. 【2017 阿里】Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction

4. 【2010】 Factorization Machines

5. https://github.com/srendle/libfm

6. https://spark.apache.org/docs/latest/ml-classification-regression.html#factorization-machines-classifier

7. https://spark.apache.org/docs/latest/ml-classification-regression.html#factorization-machines-regressor

8. https://en.wikipedia.org/wiki/Gradient_boosting

9. https://web.njit.edu/~usman/courses/cs675_fall16/BoostedTree.pdf

10. [2001] Greedy function approximation: a gradient boosting machine

11. http://www.chengli.io/tutorials/gradient_boosting.pdf

12. https://github.com/dmlc/xgboost

13. https://github.com/microsoft/LightGBM

14. https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier

15. https://spark.apache.org/docs/latest/mllib-ensembles.html#gradient-boosted-trees-gbts

16. https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html

17. https://github.com/microsoft/SynapseML

18. 【2014 GBDT+LR】Practical Lessons from Predicting Clicks on Ads at Facebook

    