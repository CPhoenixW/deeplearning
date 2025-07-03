package alexnet.network;

import alexnet.core.Tensor;

public interface Layer {
    /**
     * 前向传播：输入张量 -> 输出张量
     */
    Tensor forward(Tensor input);

    /**
     * 反向传播：从下一层来的梯度，更新本层权重并返回对前一层的梯度
     *
     * @param gradOutput 从下一层来的梯度
     * @param learningRate 学习率（用于参数更新）
     * @return 输入层的梯度（传给前一层）
     */
    Tensor backward(Tensor gradOutput, double learningRate);
}
