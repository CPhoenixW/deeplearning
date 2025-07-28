package gan.layers;

import gan.core.*;

public class SigmoidLayer implements Layer {
    private Tensor output;

    @Override
    public Tensor forward(Tensor input) {
        output = MatrixUtils.applyFunction(input, x -> 1.0 / (1.0 + Math.exp(-x)));
        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        return MatrixUtils.multiplyElementwise(gradOutput,
                MatrixUtils.applyFunction(output, x -> x * (1 - x)));
    }
}
