function net=LC_CNN(varargin)


opts.wSize=3;
opts=vl_argparse(opts,varargin);


net.layers = {} ;

lr=ones(1,10); 
weightDecay=[1 0];
w = 10 * randn(opts.wSize,opts.wSize, 1) ;  %最后一个表示输出只有一个通道
w = single(w - mean(w(:))) ;
b = single(0) ;
pad1 = ([size(w,1) size(w,1) size(w,2) size(w,2)] - 1) / 2 ;
net.layers{end+1}= struct('type', 'conv', ...
    'weights', {{w, b}}, ...
    'learningRate', [1 2], ...  %自定义learning rate相对于全局的learning rate 的比率,[1 2] 表示w的学习率与全局学习率一致，而偏置b的学习率则为全局学习率的两倍
    'weightDecay',weightDecay,...;%自定义weightDecay相对于全局的weightDecay的比率，[1 0]表示只对w矩阵进行weight decay，而不对偏置b进行weight decay
    'stride', 1, ...
    'pad', pad1) ;

rho2 = 3 ;
pad2 = (rho2 - 1) / 2 ;
net.layers{end+1}= struct('type', 'pool', ...
    'method', 'max', ...
    'pool', rho2, ...
    'stride', 1, ...
    'pad', pad2) ;



%% 自定义的误差函数层
net.layers{end+1}= struct('type', 'custom', ...
    'forward', @forwardFunc , ...  %
    'backward', @backwardFunc) ;
end


function resIp1= forwardFunc(layer, resI, resIp1)   % 计算数据项误差
pos=layer.class==1;
neg=layer.class==-1;
resIp1.x=  mean(max(0, 1 - resI.x(pos))) + ...
    mean(max(0, resI.x(neg)));
end

%后向传播，，求导；；；注意layer.class为该batch内图片的labels。
function resI = backwardFunc(layer, resI, resIp1)  %计算导数

pos=layer.class==1;
neg=layer.class==-1;
resI.dzdx=- single(resI.x < 1 & pos) / sum(pos(:)) + ...
    + single(resI.x > 0 & neg) / sum(neg(:)) ;
end
