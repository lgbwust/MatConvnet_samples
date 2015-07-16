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


%%  ת��Ϊimdb��ʽ

imdb.images.data = {im} ;
imdb.images.labels = {y} ; %labels�����ݽṹ��ȫ���Լ���������getBatch�����Լ������backward�������ݼ��ɣ���backward��������layer.class��ʽ����,layer.class����getBatch�����ķ���ֵlabels�������ڷ������⣬һ��Ϊ[1,1,2,2]�������Ϣ
imdb.images.set = [1];%  �������ͣ�1Ϊѵ��������2Ϊvalidation������3Ϊ��������,,����cnn_train����ֻ���ѵ�������Ͳ����������м���
imdb.meta.sets = {'train', 'val', 'test'} ;%�������ƣ���1����ѵ������2����val����validation��
imdb.meta.classes ={'1'}; %�������




%% -------------------------------------------------------------------------
% Part 3.3: Learning with stochastic gradient descent
% -------------------------------------------------------------------------

trainOpts.batchSize = 1 ;
trainOpts.numEpochs = 500 ; %���ж�����
trainOpts.continue = false ;%���trainOpts.expDir�Ѿ��ж�Ӧepoch�����磬�Ƿ�ֱ������
trainOpts.gpus=[]; %gpu�±꣬��ʹ��GPU����Ϊ[],,���������GPU������Ϊ [1 2]
trainOpts.learningRate =.5 ;
trainOpts.expDir = 'data/ex3-experiment' ; %�������������ʵ����ͼ�����Ķ�
trainOpts.momentum=0.9;
trainOpts.weightDecay = 0.0001 ;

trainOpts.errorFunction=@error_ErrPixels; %������ʾ�����Զ��庯��
trainOpts.errorLabels ={'������'};

net=LC_CNN('wSize',wSize); %��pos,neg���������磬��¼�������Ľṹ�У����ڸ���֮ǰ�Ĺ�ʽ�������





trainOpts.val=   imdb.images.set==1;  %��֤���±꣬����������֤�������򱨴�����ֻ�ð�ѵ����������֤����
[net, info] = cnn_train(net, imdb, @getBatch, ...
    trainOpts) ;

end


%ֻ��һ��ͼƬ��ֱ�ӷ��ؾ�����
function  [im, labels]=getBatch(imdb,batch)
   im= imdb.images.data{1};
   im=reshape(im,[size(im) 1]);
   labels=imdb.images.labels{1};
end


%��ͼ��,���ڼ�ⲻͬepoch����������ܣ����ﷵ�ص�ָ��Ϊ���������
function err = error_ErrPixels(opts, labels, res)
% -------------------------------------------------------------------------
resX=res(end-1).x;
y=labels;
        fp = resX >0 & y < 0 ;
        fn = resX < 1 & y > 0 ;
        tn = resX <= 0 & y < 0 ;
        tp = resX >= 1 & y > 0 ;
% err=[fp;fn;tn;tp]; %�������ö��ָ�꣬��Ӧ��trainOpts.errorLabels ҲҪ��Ϊ�����ǩ
err=100*(1-sum(sum(tp|tn))/sum(sum(fp|fn|tp|tn)));
end

