package alexnet.loss;

import alexnet.core.Tensor;

public class CrossEntropyLoss {
    private float lastLoss;

    /**
     * 计算交叉熵损失
     * @param predicted softmax 输出，shape: [1][1][num_classes]
     * @param targetOneHot one-hot 标签，shape: [1][1][num_classes]
     * @return 标量损失值
     */
    public float forward(Tensor predicted, Tensor targetOneHot) {
        float[] probs = predicted.getData()[0][0];
        float[] target = targetOneHot.getData()[0][0];

        float epsilon = 1e-10f;  // 防止 log(0)
        float loss = 0;

        for (int i = 0; i < probs.length; i++) {
            loss -= target[i] * Math.log(probs[i] + epsilon);
        }

        this.lastLoss = loss;
        return loss;
    }

    /**
     * 返回 softmax 的梯度：∇L/∇z = softmax_output - one_hot
     * @param predicted softmax 输出
     * @param targetOneHot one-hot 标签
     * @return 梯度张量，与 predicted 形状相同
     */
    public Tensor backward(Tensor predicted, Tensor targetOneHot) {
        float[] probs = predicted.getData()[0][0];
        float[] target = targetOneHot.getData()[0][0];
        float[] grad = new float[probs.length];

        for (int i = 0; i < probs.length; i++) {
            grad[i] = probs[i] - target[i];  // softmax_output - one_hot
        }

        float[][][] gradData = new float[1][1][grad.length];
        gradData[0][0] = grad;
        return new Tensor(gradData);
    }

    public float getLastLoss() {
        return lastLoss;
    }
}
