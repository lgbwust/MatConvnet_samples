function net=LC_CNN(varargin)


opts.wSize=3;
opts=vl_argparse(opts,varargin);


net.layers = {} ;

lr=ones(1,10); 
weightDecay=[1 0];
w = 10 * randn(opts.wSize,opts.wSize, 1) ;  %���һ����ʾ���ֻ��һ��ͨ��
w = single(w - mean(w(:))) ;
b = single(0) ;
pad1 = ([size(w,1) size(w,1) size(w,2) size(w,2)] - 1) / 2 ;
net.layers{end+1}= struct('type', 'conv', ...
    'weights', {{w, b}}, ...
    'learningRate', [1 2], ...  %�Զ���learning rate�����ȫ�ֵ�learning rate �ı���,[1 2] ��ʾw��ѧϰ����ȫ��ѧϰ��һ�£���ƫ��b��ѧϰ����Ϊȫ��ѧϰ�ʵ�����
    'weightDecay',weightDecay,...;%�Զ���weightDecay�����ȫ�ֵ�weightDecay�ı��ʣ�[1 0]��ʾֻ��w�������weight decay��������ƫ��b����weight decay
    'stride', 1, ...
    'pad', pad1) ;

rho2 = 3 ;
pad2 = (rho2 - 1) / 2 ;
net.layers{end+1}= struct('type', 'pool', ...
    'method', 'max', ...
    'pool', rho2, ...
    'stride', 1, ...
    'pad', pad2) ;



%% �Զ����������
net.layers{end+1}= struct('type', 'custom', ...
    'forward', @forwardFunc , ...  %
    'backward', @backwardFunc) ;
end


function resIp1= forwardFunc(layer, resI, resIp1)   % �������������
pos=layer.class==1;
neg=layer.class==-1;
resIp1.x=  mean(max(0, 1 - resI.x(pos))) + ...
    mean(max(0, resI.x(neg)));
end

%���򴫲������󵼣�����ע��layer.classΪ��batch��ͼƬ��labels��
function resI = backwardFunc(layer, resI, resIp1)  %���㵼��

pos=layer.class==1;
neg=layer.class==-1;
resI.dzdx=- single(resI.x < 1 & pos) / sum(pos(:)) + ...
    + single(resI.x > 0 & neg) / sum(neg(:)) ;
end
