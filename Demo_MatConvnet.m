function Demo_MatConvnet

clc
vl_setup;
vl_setupnn;
%%

%doSmooth=false;
%subMean=false;

doSmooth=true;
subMean=true;


wSize=9;
% -------------------------------------------------------------------------
% Part 3.1: Load an example image and generate its labels
% -------------------------------------------------------------------------
%%
% Load an image
im = rgb2gray(im2single(imread('peppers.png'))) ;

% Compute the location of black blobs in the image
[pos,neg] = extractBlackBlobs(im) ;

%%
% -------------------------------------------------------------------------
% Part 3.2: Image preprocessing
% -------------------------------------------------------------------------

% Pre-smooth the image
if(doSmooth)
    im = vl_imsmooth(im,3) ;
end
%%
% Subtract median value
if(subMean)
    im = im - median(im(:)) ;
end



%% Create pixel-level labels to compute the loss
y = zeros(size(pos),'single') ;
y(pos) = +1 ;
y(neg) = -1 ;


%%  转换为imdb格式

imdb.images.data = {im} ;
imdb.images.labels = {y} ; %labels的数据结构完全由自己决定，与getBatch函数以及误差层的backward函数兼容即可（在backward函数中以layer.class形式出现,layer.class等于getBatch函数的返回值labels）；对于分类问题，一般为[1,1,2,2]等类别信息
imdb.images.set = [1];%  样本类型，1为训练样本，2为validation样本，3为测试样本,,但是cnn_train函数只会对训练样本和测试样本进行计算
imdb.meta.sets = {'train', 'val', 'test'} ;%集合名称，即1代表训练集，2代表val集（validation）
imdb.meta.classes ={'1'}; %类别名称




%% -------------------------------------------------------------------------
% Part 3.3: Learning with stochastic gradient descent
% -------------------------------------------------------------------------

trainOpts.batchSize = 1 ;
trainOpts.numEpochs = 500 ; %运行多少轮
trainOpts.continue = false ;%如果trainOpts.expDir已经有对应epoch的网络，是否直接载入
trainOpts.gpus=[]; %gpu下标，不使用GPU则设为[],,如果有两个GPU可以设为 [1 2]
trainOpts.learningRate =.5 ;
trainOpts.expDir = 'data/ex3-experiment' ; %输出的网络结果、实验结果图存在哪儿
trainOpts.momentum=0.9;
trainOpts.weightDecay = 0.0001 ;

trainOpts.errorFunction=@error_ErrPixels; %用于显示误差的自定义函数
trainOpts.errorLabels ={'错误率'};

net=LC_CNN('wSize',wSize); %将pos,neg均传给网络，记录到输出层的结构中，便于根据之前的公式计算误差





trainOpts.val=   imdb.images.set==1;  %验证集下标，，必须有验证集，否则报错。。。只好把训练集当成验证集了
[net, info] = cnn_train(net, imdb, @getBatch, ...
    trainOpts) ;

end


%只有一幅图片，直接返回就行了
function  [im, labels]=getBatch(imdb,batch)
   im= imdb.images.data{1};
   im=reshape(im,[size(im) 1]);
   labels=imdb.images.labels{1};
end


%绘图用,用于检测不同epoch下网络的性能，这里返回的指标为分类错误率
function err = error_ErrPixels(opts, labels, res)
% -------------------------------------------------------------------------
resX=res(end-1).x;
y=labels;
        fp = resX >0 & y < 0 ;
        fn = resX < 1 & y > 0 ;
        tn = resX <= 0 & y < 0 ;
        tp = resX >= 1 & y > 0 ;
% err=[fp;fn;tn;tp]; %可以设置多个指标，对应的trainOpts.errorLabels 也要改为多个标签
err=100*(1-sum(sum(tp|tn))/sum(sum(fp|fn|tp|tn)));
end

