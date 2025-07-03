package alexnet.layers;

import alexnet.core.Tensor;
import alexnet.network.Layer;


public class MaxPoolingLayer implements Layer {
    private int poolSize;
    private int stride;
    private int inputHeight;
    private int inputWidth;

    private int[][][][] maxIndexes;

    public MaxPoolingLayer(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }

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
        maxIndexes = new int[channels][outH][outW][2];

        for (int c = 0; c < channels; c++) {
            for (int oH = 0; oH < outH; oH++) {
                for (int oW = 0; oW < outW; oW++) {
                    int maxH = -1;
                    int maxW = -1;
                    float maxData = Float.NEGATIVE_INFINITY;

                    for (int pH = 0; pH < poolSize; pH++) {
                        for (int pW = 0; pW < poolSize; pW++) {
                            int iH = oH * stride + pH;
                            int iW = oW * stride + pW;
                            if (inData[c][oH][oW] > maxData) {
                                maxData = inData[c][iH][iW];
                                maxH = iH;
                                maxW = iW;
                            }
                        }
                    }
                    outData[c][oH][oW] = maxData;
                    maxIndexes[c][oH][oW][0] = maxH;
                    maxIndexes[c][oH][oW][1] = maxW;
                }
            }
        }
        return new Tensor(outData);
    }
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
