package alexnet.layers;

import alexnet.network.Layer;
import alexnet.core.Tensor;

import java.util.Random;

public class ConvLayer2 implements Layer{
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

    public ConvLayer2(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        weights = new float[inChannels][outChannels][kernelSize][stride];
        biases = new float[inChannels];

        // 标准Xavier初始化。
        Random rand = new Random();
        float limit = (float) Math.sqrt(6.0 / (inChannels * kernelSize * kernelSize + outChannels));
        for (int oc = 0; oc < outChannels; oc++) {
            for (int ic = 0; ic < inChannels; ic++) {
                for (int kh = 0; kh < kernelSize; kh++) {
                    for (int kw = 0; kw < stride; kw++) {
                        weights[oc][ic][kh][kw] = (rand.nextFloat() * 2 - 1) * limit;
                    }
                }
            }
        }
    }
    @Override
    public Tensor forward(Tensor input){
        this.input = input;

        int iH = input.getHeight();
        int iW = input.getWidth();

        int oH = (iH - kernelSize + 2 * padding) / stride + 1;
        int oW = (iW - kernelSize + 2 * padding) / stride + 1;


        Tensor output = new Tensor(outChannels, oH, oW);
        float[][][] inData = input.getData();


        for(int oc = 0; oc < inChannels; oc++){
            for(int oh = 0; oh < oH; oh++){
                for(int ow = 0; ow < oW; ow++){
                    float sum = biases[oc];
                    for(int ic = 0; ic < iH; ic++){
                        for(int kh = 0; kh < kernelSize; kh++){
                            for (int kw = 0; kw < kernelSize; kw++){
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && iw >= 0 && ih < iH && iw < iW) {
                                    sum += weights[oc][ic][kh][kw] * inData[ic][ih][iw];
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
    public Tensor backward(Tensor gradOutput, double learningRate){
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
