package alexnet.network;

import alexnet.core.Tensor;
import alexnet.layers.*;

public class AlexNet {
    // 层对象声明
    private ConvLayer conv1;
    private ReluLayer relu1;
    private PoolingLayer pool1;

    private ConvLayer conv2;
    private ReluLayer relu2;
    private PoolingLayer pool2;

    private ConvLayer conv3;
    private ReluLayer relu3;

    private ConvLayer conv4;
    private ReluLayer relu4;

    private ConvLayer conv5;
    private ReluLayer relu5;
    private PoolingLayer pool5;

    private FlattenLayer flatten;

    private FCLayer fc6;
    private ReluLayer relu6;

    private FCLayer fc7;
    private ReluLayer relu7;

    private FCLayer fc8; // 输出1000类

    private SoftmaxLayer softmax;

    public AlexNet() {
        // 初始化层，参数根据AlexNet论文或者你自己设置
        conv1 = new ConvLayer(3, 96, 11, 4, 2);
        relu1 = new ReluLayer();
        pool1 = new PoolingLayer(3, 2);

        conv2 = new ConvLayer(96, 256, 5, 1, 2);
        relu2 = new ReluLayer();
        pool2 = new PoolingLayer(3, 2);

        conv3 = new ConvLayer(256, 384, 3, 1, 1);
        relu3 = new ReluLayer();

        conv4 = new ConvLayer(384, 384, 3, 1, 1);
        relu4 = new ReluLayer();

        conv5 = new ConvLayer(384, 256, 3, 1, 1);
        relu5 = new ReluLayer();
        pool5 = new PoolingLayer(3, 2);

        flatten = new FlattenLayer();

        fc6 = new FCLayer(256 * 6 * 6, 4096);
        relu6 = new ReluLayer();

        fc7 = new FCLayer(4096, 4096);
        relu7 = new ReluLayer();

        fc8 = new FCLayer(4096, 200);

        softmax = new SoftmaxLayer();
    }

    // 前向传播
    public Tensor forward(Tensor input) {
        Tensor x = conv1.forward(input);
        x = relu1.forward(x);
        x = pool1.forward(x);

        x = conv2.forward(x);
        x = relu2.forward(x);
        x = pool2.forward(x);

        x = conv3.forward(x);
        x = relu3.forward(x);

        x = conv4.forward(x);
        x = relu4.forward(x);

        x = conv5.forward(x);
        x = relu5.forward(x);
        x = pool5.forward(x);

        x = flatten.forward(x);

        x = fc6.forward(x);
        x = relu6.forward(x);

        x = fc7.forward(x);
        x = relu7.forward(x);

        x = fc8.forward(x);

        x = softmax.forward(x);

        return x;
    }

    // 反向传播，lossGrad 是最后的梯度（通常由CrossEntropyLoss给出）
    public void backward(Tensor lossGrad, double learningRate) {
        Tensor grad = softmax.backward(lossGrad, learningRate);

        grad = fc8.backward(grad, learningRate);

        grad = relu7.backward(grad, learningRate);
        grad = fc7.backward(grad, learningRate);

        grad = relu6.backward(grad, learningRate);
        grad = fc6.backward(grad, learningRate);

        grad = flatten.backward(grad, learningRate);

        grad = pool5.backward(grad, learningRate);
        grad = relu5.backward(grad, learningRate);
        grad = conv5.backward(grad, learningRate);

        grad = relu4.backward(grad, learningRate);
        grad = conv4.backward(grad, learningRate);

        grad = relu3.backward(grad, learningRate);
        grad = conv3.backward(grad, learningRate);

        grad = pool2.backward(grad, learningRate);
        grad = relu2.backward(grad, learningRate);
        grad = conv2.backward(grad, learningRate);

        grad = pool1.backward(grad, learningRate);
        grad = relu1.backward(grad, learningRate);
        grad = conv1.backward(grad, learningRate);
    }
}
