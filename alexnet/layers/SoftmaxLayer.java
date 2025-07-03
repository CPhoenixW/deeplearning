package alexnet.layers;

import alexnet.core.Tensor;
import alexnet.network.Layer;

public class SoftmaxLayer implements Layer {
    private Tensor output; // 缓存 softmax 结果用于反向传播

    @Override
    public Tensor forward(Tensor input) {
        float[] logits = input.getData()[0][0]; // 假设输入为 [1][1][N]
        float[] probs = new float[logits.length];

        // 数值稳定性处理：减去最大值
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float v : logits) {
            if (v > maxLogit) maxLogit = v;
        }

        float sumExp = 0;
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - maxLogit);
            sumExp += probs[i];
        }

        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sumExp;
        }

        float[][][] outData = new float[1][1][probs.length];
        outData[0][0] = probs;
        output = new Tensor(outData);
        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        // gradOutput 是交叉熵传来的 ∇L/∇softmax_output
        // 通常为 softmax_output - one_hot(target)
        return gradOutput;
    }
}
