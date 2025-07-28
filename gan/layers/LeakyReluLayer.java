package gan.layers;

import gan.core.*;

public class LeakyReluLayer implements Layer {
    private final double alpha = 0.01;
    private Tensor input;

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        return MatrixUtils.applyFunction(input, x -> x > 0 ? x : alpha * x);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        return MatrixUtils.multiplyElementwise(gradOutput,
                MatrixUtils.applyFunction(input, x -> x > 0 ? 1.0 : alpha));
    }
}
