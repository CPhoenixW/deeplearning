package alexnet.core;

import java.util.Random;

public class Initializer {
    private static final Random random = new Random();

    /**
     * Xavier 均匀初始化（Glorot）
     * @param fanIn 输入通道数
     * @param fanOut 输出通道数
     * @param weights 权重数组，修改原数组
     */
    public static void xavierUniform(float[][] weights, int fanIn, int fanOut) {
        float limit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = uniform(-limit, limit);
            }
        }
    }

    /**
     * He 正态初始化（适用于 ReLU 激活）
     * @param fanIn 输入通道数
     * @param weights 权重数组，修改原数组
     */
    public static void heNormal(float[][] weights, int fanIn) {
        double std = Math.sqrt(2.0 / fanIn);
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = (float) (random.nextGaussian() * std);
            }
        }
    }

    /**
     * 均匀分布生成
     */
    private static float uniform(float min, float max) {
        return min + random.nextFloat() * (max - min);
    }

    /**
     * 初始化偏置为0
     */
    public static void zeros(float[] biases) {
        for (int i = 0; i < biases.length; i++) {
            biases[i] = 0;
        }
    }
}
