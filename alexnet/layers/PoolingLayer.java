package alexnet.layers;

import alexnet.core.Tensor;
import alexnet.network.Layer;

public class PoolingLayer implements Layer {
    private int poolSize;
    private int stride;
    private int inputHeight;
    private int inputWidth;

    // 缓存最大位置用于反向传播
    private int[][][][] maxIndexes;

    public PoolingLayer(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    @Override
    public Tensor forward(Tensor input) {
        int channels = input.getChannels();
        int inH = input.getHeight();
        int inW = input.getWidth();
        this.inputHeight = inH;
        this.inputWidth = inW;

        int outH = (inH - poolSize) / stride + 1;
        int outW = (inW - poolSize) / stride + 1;

        float[][][] inData = input.getData();
        float[][][] outData = new float[channels][outH][outW];
        maxIndexes = new int[channels][outH][outW][2]; // 保存最大值的位置 [h][w]

        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float maxVal = Float.NEGATIVE_INFINITY;
                    int maxH = -1, maxW = -1;

                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int ih = oh * stride + ph;
                            int iw = oh * stride + pw;

                            if (ih < inH && iw < inW) {
                                float val = inData[c][ih][iw];
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxH = ih;
                                    maxW = iw;
                                }
                            }
                        }
                    }

                    outData[c][oh][ow] = maxVal;
                    maxIndexes[c][oh][ow][0] = maxH;
                    maxIndexes[c][oh][ow][1] = maxW;
                }
            }
        }

        return new Tensor(outData);
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        int channels = gradOutput.getChannels();
        int outH = gradOutput.getHeight();
        int outW = gradOutput.getWidth();

        float[][][] gradOutData = gradOutput.getData();

        // ✅ 用 forward 缓存的 input 尺寸，保证与 ReLU 输入一致
        float[][][] gradInput = new float[channels][inputHeight][inputWidth];

        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int ih = maxIndexes[c][oh][ow][0];
                    int iw = maxIndexes[c][oh][ow][1];
                    gradInput[c][ih][iw] += gradOutData[c][oh][ow];
                }
            }
        }

        return new Tensor(gradInput);
    }

}
