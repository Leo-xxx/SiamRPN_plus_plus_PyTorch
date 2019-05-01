
![人体目标分析与理解](https://pic2.zhimg.com/4b70deef7_is.jpg)

首发于[人体目标分析与理解](https://zhuanlan.zhihu.com/visualunderstanding)







# 视觉目标跟踪之SiamRPN++

[![lbvic](https://pic1.zhimg.com/v2-94ea003e311bfc6b0b87efdf1d7a0562_xs.jpg)](https://www.zhihu.com/people/lbvic-4)

[lbvic](https://www.zhihu.com/people/lbvic-4)



117 人赞同了该文章

## SiamRPN++



更新一下：SiamRPN++已经被CVPR2019接收，并且是oral。今年跟踪中已知的除了martin大佬的ATOM和LaSOT这个数据集以外，有5篇Siamese系列的文章，基本Siamese系列已经占了主导地位。关于这几篇的介绍可以看 

[@Qiang Wang](https://www.zhihu.com/people/2d6e027e5d25744f64f437b50df5dea1)

 

的

SiamMask

。



------

大家好，我是李搏，先预祝大家春节快乐！接下来要介绍一下我们最新的工作，DaSiamRPN的升级版：SiamRPN++，目前在多个跟踪数据集上都是state-of-the-art。

- [项目地址](https://zhuanlan.zhihu.com/p/56254712/bo-li.info/SiamRPN)
- [论文地址](https://link.zhihu.com/?target=http%3A//arxiv.org/pdf/1812.11703.pdf)

------

正文开始之前，先打一个广告：

目前组内（**商汤科技智能视频新业务算法组**， leader是 

[@武伟](https://www.zhihu.com/people/5a5edfbdd54fafad9877b3a8d8129a08)

 

）有比较多职位有空缺，包括研究员和算法工程师与实习生。具体需求会列在下面，有意向的大佬可以发简历到 libo@sensetime.com。



## 多目标跟踪

1. 对跟踪或者检测有研究经验。
2. 有扎实的代码能力，熟练掌握python和C++，熟练掌握常用深度学习框架。
3. 有CV顶会publication或者相关落地经验的优先

## 零售方向

1. 对货物识别计数分析等有相关研究经验的优先。
2. 对多模态信息聚类有相关研究经验的优先。
3. 有扎实的代码能力，熟练掌握python和C++，熟练掌握常用深度学习框架。

如果是实习的话可以适当放低要求，但一定要有上进心，代码能力较强。

同时其他方向也有不少需求，如之前 

[@MM Wang](https://www.zhihu.com/people/9443facca087fcb35cb2a2a9225ad903)

 

的动作识别等，有意向的可以考虑一下。



------

下面开始正文 (欢迎转载，注明原始地址即可)

## 一、回顾一下SiamRPN，DaSiamRPN

介绍SiamRPN++之前，我们先简单回顾一下之前的SiamRPN和DaSiamRPN

## 1. [SiamRPN](https://zhuanlan.zhihu.com/p/56254712/bo-li.info/SiamRPN)

- 基于[SiamFC](https://link.zhihu.com/?target=http%3A//www.robots.ox.ac.uk/~luca/siamese-fc.html), 引入了Faster RCNN中的RPN模块，让tracker可以回归位置、形状，可以省掉多尺度测试，进一步提高性能并加速。

![img](https://pic1.zhimg.com/80/v2-028be1d714f9e2b29617b3433a0110dc_hd.jpg)



- 这里解释一下correlation做法（涉及到后续改进）：以分类分支为例，在RPN中，分类分支需要输出一个通道数为 ![2k](https://www.zhihu.com/equation?tex=2k) 的特征图（ ![k](https://www.zhihu.com/equation?tex=k) 为anchor个数），SiamFC中使用的correlation只能提供通道数为1的响应图，无法满足要求。所以我们换了一个想法，把correlation层当成一个卷积层，template分支提取的特征作为卷积核，detection分支提取的特征作为卷积层的input，这样只需要改变卷积核的形状就可以达到输出2k通道数的目的。具体做法为使用了两个不同的卷积层，template分支的卷积层负责升维，把通道数提升到 ![256*2k](https://www.zhihu.com/equation?tex=256%2A2k) ，为了保持对齐，detection分支也增加了一个卷积层，不过保持通道数不变。之后进行correlation操作（卷积），得到最终的分类结果。

## 2. [DaSiamRPN](https://zhuanlan.zhihu.com/p/56254712/bo-li.info/DaSiamRPN)

- 基于SiamRPN，主要是提出更好的使用数据，利用更好的训练方式让tracker变得更鲁邦
- 有了更好的分数作为指导后，算法可以扩展到Long-term

## 3. VOT2018 entry SiamRPN

- 基于DaSiamRPN，想通过增强backbone来提升性能，但没能实现，最终通过加厚alexnet提升性能。加厚虽然临时的提升了网络的性能，但网络依然很浅，同时还极大的增加了模型参数量以及计算量。（模型加厚一倍，实际上模型和计算量增加四倍）

## 二、SiamRPN++

## motivation

这次的motivation就是解决网络问题。从SiamFC以来，改进很多，但是网络还一直都是AlexNet，换了深网络后性能没上升不说，反而带来了极大的下降。[[github issue](https://link.zhihu.com/?target=https%3A//github.com/bertinetto/siamese-fc/issues/37%23issue-271271390)]

## 问题分析

首先回顾SiamFC, 通过相关操作，可以考虑成滑窗的形式计算每个位置的相似度。

![ f(\mathbf{z}, \mathbf{x})=\phi(\mathbf{z})\ast \phi(\mathbf{x})+b ](https://www.zhihu.com/equation?tex=+f%28%5Cmathbf%7Bz%7D%2C+%5Cmathbf%7Bx%7D%29%3D%5Cphi%28%5Cmathbf%7Bz%7D%29%5Cast+%5Cphi%28%5Cmathbf%7Bx%7D%29%2Bb+)

这里带来了两个限制：

- 网络需要满足严格的平移不变性。如SiamFC中介绍的，padding会破坏这种性质。
- 网络有对称性，即如果将搜索区域图像和模板区域图像，输出的结果应该不变。（因为是相似度，所以应该有对称性）。

对于这两点：

- 现代化网络：随着kaiming he的残差网络的提出，网络的深度得到了巨大的释放，通常物体检测和语义分割的baseline backbone都采用ResNet50的结构。为了保证网络具有适当/整齐的分辨率，几乎所有的现代网络backbone都具有padding结构。(可以参考一下这个讨论中[[padding or not padding](https://link.zhihu.com/?target=https%3A//stats.stackexchange.com/questions/246512/convolutional-layers-to-pad-or-not-to-pad)]的第三小点) 如ResNet肯定不具备严格平移不变性，padding的引入使得网络输出的响应对不同位置有了不同的认知。而我们在这一步的训练希望的是网络学习到如何**通过表观**来分辨回归物体，这里就限制了深网络在tracking领域的应用。
- 网络对称性：由于SiamRPN的监督不再是相似度，而是回归的偏移量/前背景分数，不再具有对称性。所以在SiamRPN的改进中需要引入**非对称**的部件（adj_x in Fig. 3），如果完全Siamese的话没法达到目的。这一点主要会引导后面的correlation设计。

我们认为现代化网络破坏严格平移不变性以后，带来的弊端就是会学习到位置偏见：按照SiamFC的训练方法，正样本都在正中心，网络会学到这种统计特性，学到样本中正样本分布的情况。

![img](https://pic1.zhimg.com/80/v2-8ec6210bdd5c7d1db7f47c22ee2d71f8_hd.jpg)



为了验证我们的猜测，我们设计了一个模拟实验。当我们像SiamFC一样训练，把正样本都放在中心时，网络只对中心有响应；如果把正样本均匀分布在某个范围内，而不是一直在中心时（范围是离中心点一定距离，该距离为shift；正样本在这个范围内均匀分布），随着shift的不断增大，这种现象能够逐渐得到缓解。

![img](https://pic1.zhimg.com/80/v2-c621f6ab6a9b6a15834457b2c2049c1c_hd.jpg)



我们按照这个思想进行了实际的实验验证，在训练过程中，我们不再把正样本放在中心，而是**以均匀分布的采样方式让目标在中心点附近进行偏移**。由上图可以看出，随着偏移的范围增大，深度网络可以由刚开始的完全没有效果逐渐变好。

所以说，通过均匀分布的采样方式让目标在中心点附近进行偏移，可以缓解网络因为破坏了严格平移不变性带来的影响，即消除了位置偏见，让现代化网络可以应用于跟踪中。

## 额外讨论：

**为什么你所说的问题在检测和语义分割中并不存在？** 因为对于物体检测和语义分割而言，训练过程中，物体本身就是在全图的每个位置较为均匀的分布。我们可以很容易的验证，如果在物体检测网络只训练标注在图像中心的样本，而边缘的样本都不进行训练，那么显然，这样训练的网络只会对图像的中心位置产生高响应，边缘位置就随缘了，不难想象这种时候边缘位置的性能显然会大幅衰减。而更为致命的是，按照SiamFC的训练方式，中心位置为正样本，边缘位置为负样本。那么网络只会记录下边缘永远为负，不管表观是什么样子了。这完全背离了我们训练的初衷。

## 在跟踪中使用深网络

![img](https://pic2.zhimg.com/80/v2-ece4cc7ef472a828b24255a243ff1b75_hd.jpg)



我们主要的实验实在ResNet-50上做的。现代化网络一般都是stride32，但跟踪为了定位的准确性，一般stride都比较小（Siamese系列一般都为8），所以我们把ResNet最后两个block的stride去掉了，同时增加了dilated convolution，一是为了增加感受野，二是为了能利用上预训练参数。论文中提到的MobileNet等现代化网络也是进行了这样的改动。如上图所示，改过之后，后面三个block的分辨率就一致了。

在训练过程中采用了新的采样策略后，我们可以训练ResNet网络了，并且能够正常跟踪一些视频了。（之前跟踪过程中一直聚集在中心，根本无法正常跟踪目标）。对backbone进行finetune以后，又能够进一步得到一些性能提升。

## 多层融合

加上了现代化网络以后，一个自然的想法就是使用多层融合。我们选择了网络最后三个block的输出进行融合（由于之前对网络的改动，所以分辨率一致，融合时实现起来简单）。对于融合方式上我们并没有做过多的探究，而是直接做了线性加权。

![ \mathcal{S}{all} = \sum_{l=3}^{5}\alpha_i\mathcal{S}{l},\quad \mathcal{B}{all} = \sum_{l=3}^{5}\beta_i\mathcal{B}_{l}. ](https://www.zhihu.com/equation?tex=+%5Cmathcal%7BS%7D%7Ball%7D+%3D+%5Csum_%7Bl%3D3%7D%5E%7B5%7D%5Calpha_i%5Cmathcal%7BS%7D%7Bl%7D%2C%5Cquad+%5Cmathcal%7BB%7D%7Ball%7D+%3D+%5Csum_%7Bl%3D3%7D%5E%7B5%7D%5Cbeta_i%5Cmathcal%7BB%7D_%7Bl%7D.+)

## Depthwise Cross Correlation

这一点是一个通用的改进，并不是只针对于深网络的。

![img](https://pic3.zhimg.com/80/v2-1cc179a32de97dde51fb9a640838d97a_hd.jpg)



下面分别来介绍与之前的区别：

- Cross Correlation：如上图(a)，用于SiamFC，模版支特征在搜索区域上滑窗的方式获取不同位置的响应。
- Up-Channel Cross Correlation：如上图(b)，用于SiamRPN，于Cross Correlation不同的是在做correlation前多了两个卷积层，一个提升维度（通道数），另一个保持不变。之后通过卷积的方式，得到最终的输出。通过控制升维的卷积来实现最终输出特征图的通道数。
- Depthwise Cross Correlation：如上图(c)，和UpChannel一样，在做correlation操作以前，模版和搜索分支会分别过一个卷积，但不需要提升维度，这里只是为了提供一个非Siamese的特征（SiamRPN中与SiamFC不同，比如回归分支，是非对称的，因为输出不是一个响应值；需要模版分支和搜索分支关注不同的内容）。在这之后，通过类似depthwise卷积的方法，逐通道计算correlation结果，这样的好处是可以得到一个通道数非1的输出，可以在后面添加一个普通的 ![1\times1](https://www.zhihu.com/equation?tex=1%5Ctimes1) 卷积就可以得到分类和回归的结果。
- 这里的改进主要源自于upchannel的方法中，升维卷积参数量极大， ![256\times(256*2k)\times3\times3](https://www.zhihu.com/equation?tex=256%5Ctimes%28256%2A2k%29%5Ctimes3%5Ctimes3) ， 分类分支参数就有接近6M的参数，回归分支12M。其次升维造成了两支参数量的极度不平衡，模版分支是搜索支参数量的 ![2k/4k](https://www.zhihu.com/equation?tex=2k%2F4k) 倍，也造成整体难以优化，训练困难。
- 改为Depthwise版本以后，参数量能够急剧下降；同时整体训练也更为稳定，整体性能也得到了加强。

## 实验

## Ablation

为了验证我们的提出的内容，我们做了详细的对比实验

![img](https://pic1.zhimg.com/80/v2-127d2ea2fcac8f471d123bc29128ee74_hd.jpg)

- 网络方面，从AlexNet换成了ResNet50以后，我们发现只有conv4的时候就取得了非常好的效果。虽然conv3和conv5效果没有那么好，但由于鲁棒性的提升，使得后续的提升变得有可能。同时对BackBone进行finetune也能带来接近两个点的提升。
- 多支融合，可以从图中看出，同时使用三支的效果明显比单支的要高，VOT上比最好的conv4还要高4个多点。
- 最后是新的correlation方式，从表中也可以看出，无论是AlexNet还是ResNet，装备了新的correlation方式以后，都有接近两个点提升。

![img](https://pic3.zhimg.com/80/v2-1cdb76626e945562ac523df8847f9d72_hd.jpg)



- 同时，我们还用了不同的backbone验证了top1 acc和OTB性能的曲线，也证明了我们的算法能够随着backbone的提升而提升。

## state-of-the-art

为了验证我们提出的SiamRPN++的性能，我们在六个数据集上进行了实验。首先是比较重要的两个数据集，VOT和OTB，然后添加了UAV123，同时在两个比较大的数据集LaSOT，TrackingNet上也进行了实验。最后我们又将算法应用于longterm，在VOT18-LT上进行了实验。我们新提出的算法在这些数据集上都取得了非常好的效果。

## VOT数据集

![img](https://pic2.zhimg.com/80/v2-085bb2c5ca8bf0ca5d603ed4c00b21ed_hd.png)



- 在VOT2018 baseline上，我们和排名前十的算法进行了比较。可以看出，我们EAO超过了VOT2018冠军2.5个点，目前性能最高。
- 与此同时，我们还对速度进行了分析，在下面的图中，横轴是速度（FPS），纵轴是EAO，可以看出我们是实时算法中唯一一个性能超过0.4的。即使换上了ResNet50，我们还是可以在跑realtime实验时性能没有任何下降。

![img](https://pic2.zhimg.com/80/v2-4609f8ed675537ce1b1bb1480504c559_hd.jpg)



- 此外，我们还在VOT2016上测试了我们算法，性能有0.474，也是目前的最高值。

## OTB数据集

![img](https://pic1.zhimg.com/80/v2-882ba9d903bf4ee8a13eb0b22437f0e4_hd.jpg)



- OTB数据集一直是我们的软肋，之前的SiamRPN和DaSiamRPN在OTB上表现都比较一般，但这次我们达到了0.696的AUC，终于在OTB上也达到了state-of-the-art。

## UAV123



![img](https://pic3.zhimg.com/80/v2-599376106240573a54a4fd36e3cdfa42_hd.jpg)

- 由于DaSiamRPN中longterm时做了UAV123的实验，所以我们也在上面跑了一下实验，比之前也高了三个点左右。

## LaSOT

![img](https://pic4.zhimg.com/80/v2-2c90603ec09917eeb812b862fd657e87_hd.jpg)



- 为了验证算法性能，我们又跑了最近出的LaSOT，可以看出在这个数据集上我们性能提升十分显著。相对于LaSOT论文中的性能最高的tracker MDNet提升了接近10个点。

## TrackingNet

![img](https://pic1.zhimg.com/80/v2-57a94b38d2552fbb1555dc8ca1e50c7c_hd.jpg)



- 由于前面的数据集都比较小，最近又正好出了TrackingNet（test 511段视频）。所以我们也进行了评测。这里是[LeaderBoard链接](https://link.zhihu.com/?target=http%3A//eval.tracking-net.org/featured-challenges/39/leaderboard/42) 。

## VOT18-LT

![img](https://pic1.zhimg.com/80/v2-9602b34d05325fb5be4853939e958020_hd.jpg)



- 和DaSiamRPN中一样，我们将SiamRPN++也应用到了Long Term中，由于VOT18包含UAV20L的全部视频，所以我们就没有单独评测UAV20L了。从上图可以看出，我们提出的算法在Long term上也能带来2个点的提升。

## 总结

总结一下，深网络一直是Siamese系列的一大痛点，我们简单的通过调整训练过程中正样本的采样方式让深网络可以在跟踪中发挥作用，同时通过多层聚合更大程度的发挥深网络的作用。除此之外，新的轻量级的correlation方式在减少参数量的同时，也增加了跟踪的性能。最终，我们的算法可以在多个数据集上达到state of the art的性能。

编辑于 2019-03-10

目标跟踪

赞同 117

分享



### 文章被以下专栏收录

- ![人体目标分析与理解](https://pic2.zhimg.com/4b70deef7_xs.jpg)

- ## [人体目标分析与理解](https://zhuanlan.zhihu.com/visualunderstanding)

- 专注图像和视频中的人体目标分析与理解，包括目标检测，单目标跟踪，多目标跟踪，人体行为识别，姿态(关键点)估计与跟踪，行人重识别。

- 关注专栏

### 推荐阅读

- 

- # ResNet及其变种的结构梳理、有效性分析与代码解读

- Pascal

- 

- # CVPR2018视觉目标跟踪之 SiameseRPN

- 朱政

- # FCN --2015CVPR 理解和解读

- ﻿最近关于 Anchor \ Free 的目标检测结构如雨后春笋般出现，我也跟一波学习，发现,里面涉及到很多细碎的知识，没办法啊，只有一层一层拨开它的心！道阻且长啊~~~~~写的不好望指点哈！旨在…

- wzy y...发表于计算机视觉...

- 

- # 带你入门多目标跟踪（二）SORT&DeepSORT

- ZihaoZhao

## 14 条评论

切换为时间排序

写下你的评论...







- [![柳叶叶](https://pic2.zhimg.com/902632b7e_s.jpg)](https://www.zhihu.com/people/liu-xie-xie)

  [柳叶叶](https://www.zhihu.com/people/liu-xie-xie)

  2 个月前

  加厚alexnet是指增加通道数目？

  赞回复踩举报

- [![Qiang Wang](https://pic4.zhimg.com/27d64a5a37281c2f995162313c243926_s.jpg)](https://www.zhihu.com/people/qiang-wang-56-56)

  [Qiang Wang](https://www.zhihu.com/people/qiang-wang-56-56)

  回复

  [柳叶叶](https://www.zhihu.com/people/liu-xie-xie)

  2 个月前

  SiamRPN在参加VOT2018比赛的时候将alexnet backbone的每层的channel都增加了一倍。具体可以参看模型文件。
  [https://github.com/foolwood/DaSiamRPN/blob/master/code/net.py](http://link.zhihu.com/?target=https%3A//github.com/foolwood/DaSiamRPN/blob/master/code/net.py)

  2回复踩举报

- [![虚伪撕裂者](https://pic4.zhimg.com/da8e974dc_s.jpg)](https://www.zhihu.com/people/qi-cheng-zuo)

  [虚伪撕裂者](https://www.zhihu.com/people/qi-cheng-zuo)

  2 个月前

  这个在代码上的具体体现是不是就是在 siamrpn 中对 search image 做了比较大translation 的random crop

  赞回复踩举报

- [![Qiang Wang](https://pic4.zhimg.com/27d64a5a37281c2f995162313c243926_s.jpg)](https://www.zhihu.com/people/qiang-wang-56-56)

  [Qiang Wang](https://www.zhihu.com/people/qiang-wang-56-56)

  回复

  [虚伪撕裂者](https://www.zhihu.com/people/qi-cheng-zuo)

  2 个月前

  是的。

  

  理想中的做法应该是统计每种padding 模式的概率，并按照该概率的倒数归一化后算出理想的概率分布。但这种方法过于复杂。

  

  实际中测试只需要在搜索图像中将目标进行随机移动即可。

  赞回复踩举报

- [![轩爷](https://pic2.zhimg.com/v2-0b3044f01cf6337284e70d3d62adc71c_s.jpg)](https://www.zhihu.com/people/xuan-ye-97)

  [轩爷](https://www.zhihu.com/people/xuan-ye-97)

  回复

  [Qiang Wang](https://www.zhihu.com/people/qiang-wang-56-56)

  1 个月前

  请问一下如果是使用这种偏移的方法训练普通的siamfc将backbone换成具有padding的网络是否可行

  赞回复踩举报

- [![图波列夫](https://pic4.zhimg.com/da8e974dc_s.jpg)](https://www.zhihu.com/people/tu-bo-lie-fu)

  [图波列夫](https://www.zhihu.com/people/tu-bo-lie-fu)

  2 个月前

  该评论已删除

- [![lbvic](https://pic1.zhimg.com/v2-94ea003e311bfc6b0b87efdf1d7a0562_s.jpg)](https://www.zhihu.com/people/lbvic-4)

  [lbvic](https://www.zhihu.com/people/lbvic-4)

   (作者) 

  回复

  [图波列夫](https://www.zhihu.com/people/tu-bo-lie-fu)

  1 个月前

  已经修正

  赞回复踩举报

- [![YaqiLYU](https://pic4.zhimg.com/v2-1b8bc96f476d3818cd1f503c6077d68a_s.jpg)](https://www.zhihu.com/people/YaqiLYU)

  [YaqiLYU](https://www.zhihu.com/people/YaqiLYU)

  1 个月前

  这个结果太爆炸了，手动点赞

  1回复踩举报

- [![lbvic](https://pic1.zhimg.com/v2-94ea003e311bfc6b0b87efdf1d7a0562_s.jpg)](https://www.zhihu.com/people/lbvic-4)

  [lbvic](https://www.zhihu.com/people/lbvic-4)

   (作者) 

  回复

  [YaqiLYU](https://www.zhihu.com/people/YaqiLYU)

  1 个月前

  性能终于可以和CF方法抗衡啦

  赞回复踩举报

- [![YaqiLYU](https://pic4.zhimg.com/v2-1b8bc96f476d3818cd1f503c6077d68a_s.jpg)](https://www.zhihu.com/people/YaqiLYU)

  [YaqiLYU](https://www.zhihu.com/people/YaqiLYU)

  回复

  [lbvic](https://www.zhihu.com/people/lbvic-4)

   (作者)

  1 个月前

  这个已经不是抗衡了，时代都变了

  赞回复踩举报

- [![堃堃](https://pic4.zhimg.com/56c592f12_s.jpg)](https://www.zhihu.com/people/kun-kun-97-81)

  [堃堃](https://www.zhihu.com/people/kun-kun-97-81)

  1 个月前

  siamFC说网络加了padding会违一个所提出的Fully-convolutional Siamese architecture的一个公式：h(L_{k\tau}x)=L_{\tau}h(x)L_{\tau}x)[u] = x[u- \tau]， 请问这个公式该如何理解呢？

  赞回复踩举报

- [![lbvic](https://pic1.zhimg.com/v2-94ea003e311bfc6b0b87efdf1d7a0562_s.jpg)](https://www.zhihu.com/people/lbvic-4)

  [lbvic](https://www.zhihu.com/people/lbvic-4)

   (作者) 

  回复

  [堃堃](https://www.zhihu.com/people/kun-kun-97-81)

  1 个月前

  简单的说，边缘的patch会由于padding带来黑边，而靠近中心的patch则不会有。所以网络的预测本身是不满足全卷积性质的，也就是在边缘卷出来的和中心卷出来的内容是不一致的。这就可能导致训练过程中，网络倾向于通过padding的pattern来确定目标在哪，即padding少的位置接近中心，更可能是目标。

  2回复踩举报

- [![随月弄轻舞](https://pic4.zhimg.com/v2-8c24a6412e72b05cca78893119340238_s.jpg)](https://www.zhihu.com/people/sui-yue-nong-qing-wu)

  [随月弄轻舞](https://www.zhihu.com/people/sui-yue-nong-qing-wu)

  27 天前

  这里说的平移不变性是平移可变性还是不变性?

  赞回复踩举报

- [![祝中科](https://pic1.zhimg.com/v2-7e4adea1adfd237dd844851bf8679b08_s.jpg)](https://www.zhihu.com/people/zhu-zhong-ke)

  [祝中科](https://www.zhihu.com/people/zhu-zhong-ke)

  20 天前

  大神，项目地址是挂了吗，看不到了。

  赞回复踩举报

- [![Jolkin](https://pic3.zhimg.com/v2-a7d8c372e06dcef72edf9d55d5ddf7a6_s.jpg)](https://www.zhihu.com/people/jolkin)

  [Jolkin](https://www.zhihu.com/people/jolkin)

  14 天前

  CVPR19有篇oral-叫CIR专门讨论了如何加深SiamTracker网络的问题

  1回复踩举报