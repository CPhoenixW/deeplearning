package gan.layers;

import gan.core.*;

public class TanhLayer implements Layer {
    private Tensor output;

    @Override
    public Tensor forward(Tensor input) {
        output = MatrixUtils.applyFunction(input, Math::tanh);
        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        return MatrixUtils.multiplyElementwise(gradOutput,
                MatrixUtils.applyFunction(output, x -> 1 - x * x));
    }
}
