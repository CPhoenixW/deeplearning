package gan.layers;

import gan.core.*;

public class DenseLayer implements Layer {
    private final int inputSize;
    private final int outputSize;

    private Tensor weights;
    private Tensor bias;
    private Tensor input;

    private Tensor gradWeights;
    private Tensor gradBias;

    private final double learningRate = 0.01;

    public DenseLayer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = Initializer.xavier(inputSize, outputSize);
        this.bias = Initializer.zeros(1, outputSize);
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        return MatrixUtils.add(MatrixUtils.dot(input, MatrixUtils.transpose(weights)), bias);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        gradWeights = MatrixUtils.dot(MatrixUtils.transpose(gradOutput), input);
        gradBias = gradOutput;

        weights = MatrixUtils.subtract(weights, MatrixUtils.scalarMultiply(gradWeights, learningRate));
        bias = MatrixUtils.subtract(bias, MatrixUtils.scalarMultiply(gradBias, learningRate));

        return MatrixUtils.dot(gradOutput, weights);
    }
}
