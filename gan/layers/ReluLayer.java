package gan.layers;

import gan.core.*;

public class ReluLayer implements Layer {
    private Tensor input;

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        return MatrixUtils.applyFunction(input, x -> Math.max(0, x));
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        return MatrixUtils.multiplyElementwise(gradOutput, MatrixUtils.applyFunction(input, x -> x > 0 ? 1.0 : 0.0));
    }
}
