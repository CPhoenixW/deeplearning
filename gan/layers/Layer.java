package gan.layers;

import gan.core.Tensor;

public interface Layer {
    Tensor forward(Tensor input);
    Tensor backward(Tensor gradOutput);
}

