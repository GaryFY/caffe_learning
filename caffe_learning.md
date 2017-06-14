Caffe学习笔记
====

### caffe的三级结构  

1.blob：数据的保存，交换以及操作都是以blob形式进行的，数据维度为**number N × channel K × height H × width W**   
2.layer：模型和计算的基础。每一层定义三种操作，setup（Layer初始化），forward(正向传导，根据input计算output)，backward(反向传导计算，根据output计算input的梯度)    
3.net：整合并连接layer，由一系列layer组成     
模型格式：.prototxt,训练好的模型在model目录下.binnaryproto格式的文件中，模型的格式由caffe.proto定义  

#### Layer相关   
##### 五大派生类型  

1.data_layer:
* DATA:用于LevelDB和LMDB数据格式的输入的类型
* MEMORY_DATA:
* HDF5_DATA:
* HDF5_OUTPUT:
* IMAGE_DATA:图像格式数据输入的类型   
2.neuron_layer:实现大量激活函数,元素级别的运算   
3.loss_layer:计算网络误差   
4.common_layer:主要进行vision_layer的连接   
5.vision_layer:主要实现convolution和polling操作   

##### 重要成员函数与成员变量  
1.forward(正向传导，根据bottom计算top)，backward(反向传导计算，根据top计算bottom)  

2.loss

##### Solver  
当进行整个网络训练过程的时候，实际上是在运行caffe.cpp中的train()函数，而这个函数实际上是实例化一个Solver对象，初始化后调用了Solver中的Solve()方法。而这个Solve()函数主要就是在迭代运行下面这两个函数

```
ComputerUpdateValue();
net_->Update();
```

### 搭建网络（以mnist为例）  
1.数据准备库：   
```
cd $CAFFE_ROOT/data/mnist
./get_mnist.sh
cd $CAFFE_ROOT/examples/mnist
./create_mnist.sh
```
会有mnist_train_leveldb和mnist_test_leveldb两个文件夹
**若自己准备数据库,首先下载数据，分为train和test两类数据库，然后将训练样本文件名与标签列出train.txt和test.txt，分类的名字是ADCII码顺序，0-999,然后修改imagenet下的create_imagenet.sh,主要修改训练和测试路径，最后得到lmdb或leveldb数据格式，得到两个文件夹imagenet_train_leveldb和imagenet_val_leveldb**   
2.定义训练网络   
(1)lenet_train_test.prototxt      
修改
```
examples/mnist/lenet_train_test.prototxt
```
顺序：**网络命名**->**写入数据层**（名字，类型，参数：数据源/批次大小/归一化，连接data和label Blob空间）->**卷积层**（名字，类型，前为data后为conv1的Blob空间，学习率，参数：输出单元数/卷积核大小/步长/权重与偏置）->**池化层**（名字，类型，前为conv1后为pool1的Blob空间），参数（方式，核，步长，）-> **...卷积层+池化层...** -> 全连接层（名字，类型，参数：输出节点，权重与偏置，前为pooln后卫ip1）->**ReLU层**->**全连接层**->**LOSS层**（第一块是预测，第二次是数据层提供的标签）

(2)lenet_solver.prototxt  
定义训练和检测数据来源->每个批次个数和迭代次数->基础学习率，动量，权重衰减项->学习策略->显示次数->最大迭代次数->数据存储->CPU/GPU
3.训练和测试
```
./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt
```
### caffemodel可视化  
利用caffe进行训练，将训练的结果模型进行保存，得到一个caffemodel，然后从测试图片中选出一张进行测试，并进行可视化    

1.加载必要的库
```py
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import sys,os,caffe
```
2.设置当前目录，判断模型是否训练好
```py
sys.path.insert()
os.chdir()
```
3.利用提前训练好的模型，设置测试网络

4.加载测试图片，并显示

5.编写一个函数，将二进制的均值转换为python的均值
```py
#　编写一个函数，将二进制的均值转换为python的均值
def convert_mean(binMean,npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean, 'rb' ).read()
    blob.ParseFromString(bin_mean)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    npy_mean = arr[0]
    np.save(npyMean, npy_mean )
binMean=caffe_root+'examples/cifar10/mean.binaryproto'
npyMean=caffe_root+'examples/cifar10/mean.npy'
convert_mean(binMean,npyMean)
```
6.将图片载入blob中，并减去均值
```py
#将图片载入blob中,并减去均值
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(npyMean).mean(1).mean(1)) # 减去均值
transformer.set_raw_scale('data', 255)  
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].data[...] = transformer.preprocess('data',im)
inputData=net.blobs['data'].data
```
7.显示减去均值前后的数据

8.运行测试模型，并显示各层数据信息
```py
#运行测试模型，并显示各层数据信息
net.forward()
[(k, v.data.shape) for k, v in net.blobs.items()]
```
9.显示各层的参数信息
```py
[(k, v[0].data.shape) for k, v in net.params.items()]
```
10.编写一个函数，用于显示各层数据

```py
#　编写一个函数，用于显示各层数据
def show_data(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()  #对输入的图像进行narmlization
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))  #强制性地使输入的图像个数为平方数，不足平方数时，手动添加几幅
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)  #每幅小图像之间加入小空隙
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.imshow(data,cmap='gray')
    plt.axis('off')
```
11.显示第一个卷积层的输出数据和权值(filter)
```py
#显示第一个卷积层的输出数据和权值（filter）
net.blobs['layer name'].data[0] #获取网络各层的数据（输出数据）

net.params['layer name'][0].data #获取网络各层的参数数据（权值）

```
12.显示第一次pooling层后的输出数据
13.重复11,12步骤
14.输出最后一层输入属于某个类的概率

### 制作网络模型  
例：第一个参数：网络模型的prototxt文件，第二个参数：保存的图片路径及名字，第三个参数：--rankdir=x,x有四种选项，分别是LR,RL,TB,BT(网络的方向)
```
sudo python python/draw_net.py examples/cifar10_full_train_test.prototxt netImage/cifar10.png --rankdir=BT
```

### 绘制loss和accuracy曲线 
 
1.加载必要的库  
2.设置当前目录  
3.设置是solver求解器  
```py
%%time
niter =4000
test_interval = 200
train_loss = np.zeros(niter) #用于每次迭代train loss
test_acc = np.zeros(int(np.ceil(niter / test_interval))) #用于保存test accuracy

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='conv1')
    
    if it % test_interval == 0:
        acc=solver.test_nets[0].blobs['accuracy'].data
        print 'Iteration', it, 'testing...','accuracy:',acc
        test_acc[it // test_interval] = acc
```
4.绘制train过程中的loss曲线，和测试过程中的accuracy曲线
```py
print test_acc
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(niter), train_loss)
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
```
