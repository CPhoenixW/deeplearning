package alexnet.layers;

import alexnet.core.Tensor;
import alexnet.network.Layer;

import java.util.Random;

public class ConvLayer implements Layer {
    private int inChannels;
    private int outChannels;
    private int kernelSize;
    private int stride;
    private int padding;

    private float[][][][] weights;  // [outChannels][inChannels][kernelSize][kernelSize]
    private float[] biases;         // [outChannels]

    private Tensor input;           // 缓存用于反向传播
    private float[][][][] gradWeights;
    private float[] gradBiases;

    public ConvLayer(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;

        // 权重初始化 (Xavier)
        Random rand = new Random();
        float limit = (float) Math.sqrt(6.0 / (inChannels * kernelSize * kernelSize + outChannels));
        weights = new float[outChannels][inChannels][kernelSize][kernelSize];
        for (int o = 0; o < outChannels; o++) {
            for (int i = 0; i < inChannels; i++) {
                for (int kh = 0; kh < kernelSize; kh++) {
                    for (int kw = 0; kw < kernelSize; kw++) {
                        weights[o][i][kh][kw] = (rand.nextFloat() * 2 - 1) * limit;
                    }
                }
            }
        }

        biases = new float[outChannels];
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input; // 缓存输入用于反向传播

        int inH = input.getHeight();
        int inW = input.getWidth();
        int outH = (inH - kernelSize + 2 * padding) / stride + 1;
        int outW = (inW - kernelSize + 2 * padding) / stride + 1;

        Tensor output = new Tensor(outChannels, outH, outW);
        float[][][] inData = input.getData();

        for (int oc = 0; oc < outChannels; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = biases[oc];
                    for (int ic = 0; ic < inChannels; ic++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                    sum += inData[ic][ih][iw] * weights[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    output.set(oc, oh, ow, sum);
                }
            }
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        int inH = input.getHeight();
        int inW = input.getWidth();
        int outH = gradOutput.getHeight();
        int outW = gradOutput.getWidth();

        float[][][] inData = input.getData();
        float[][][] gradOut = gradOutput.getData();

        float[][][] gradInput = new float[inChannels][inH][inW];

        // 初始化梯度累积器
        gradWeights = new float[outChannels][inChannels][kernelSize][kernelSize];
        gradBiases = new float[outChannels];

        // 计算梯度 w.r.t 权重, 输入, 偏置
        for (int oc = 0; oc < outChannels; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float grad = gradOut[oc][oh][ow];
                    gradBiases[oc] += grad;
                    for (int ic = 0; ic < inChannels; ic++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                    gradWeights[oc][ic][kh][kw] += inData[ic][ih][iw] * grad;
                                    gradInput[ic][ih][iw] += weights[oc][ic][kh][kw] * grad;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 更新权重和偏置
        for (int oc = 0; oc < outChannels; oc++) {
            biases[oc] -= learningRate * gradBiases[oc];
            for (int ic = 0; ic < inChannels; ic++) {
                for (int kh = 0; kh < kernelSize; kh++) {
                    for (int kw = 0; kw < kernelSize; kw++) {
                        weights[oc][ic][kh][kw] -= learningRate * gradWeights[oc][ic][kh][kw];
                    }
                }
            }
        }

        return new Tensor(gradInput);
    }
}
