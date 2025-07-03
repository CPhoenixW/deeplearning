package alexnet.layers;

import alexnet.core.Tensor;
import alexnet.network.Layer;

import java.util.Random;

public class FCLayer implements Layer {
    private int inputSize;
    private int outputSize;

    private float[][] weights; // [outputSize][inputSize]
    private float[] biases;    // [outputSize]

    private float[] inputCache; // 缓存用于反向传播

    public FCLayer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = new float[outputSize][inputSize];
        this.biases = new float[outputSize];

        // Xavier 初始化
        Random rand = new Random();
        float limit = (float) Math.sqrt(6.0 / (inputSize + outputSize));
        for (int o = 0; o < outputSize; o++) {
            for (int i = 0; i < inputSize; i++) {
                weights[o][i] = (rand.nextFloat() * 2 - 1) * limit;
            }
            biases[o] = 0;
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        // 输入应为 [1, 1, inputSize] 扁平张量
        float[] flatInput = flatten(input);
        this.inputCache = flatInput;

        float[][][] outputData = new float[1][1][outputSize];

        for (int o = 0; o < outputSize; o++) {
            float sum = biases[o];
            for (int i = 0; i < inputSize; i++) {
                sum += weights[o][i] * flatInput[i];
            }
            outputData[0][0][o] = sum;
        }

        return new Tensor(outputData); // shape: [1][1][outputSize]
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        float[] gradOut = gradOutput.getData()[0][0]; // [outputSize]
        float[] gradInput = new float[inputSize];     // [inputSize]

        // 计算梯度 & 更新权重
        for (int o = 0; o < outputSize; o++) {
            for (int i = 0; i < inputSize; i++) {
                gradInput[i] += weights[o][i] * gradOut[o]; // 输入误差
                weights[o][i] -= learningRate * gradOut[o] * inputCache[i];
            }
            biases[o] -= learningRate * gradOut[o];
        }

        float[][][] gradInputData = new float[1][1][inputSize];
        gradInputData[0][0] = gradInput;
        return new Tensor(gradInputData);
    }

    private float[] flatten(Tensor input) {
        float[][][] data = input.getData();
        int c = input.getChannels();
        int h = input.getHeight();
        int w = input.getWidth();

        float[] flat = new float[c * h * w];
        int index = 0;
        for (int ch = 0; ch < c; ch++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    flat[index++] = data[ch][i][j];
                }
            }
        }
        return flat;
    }
}
