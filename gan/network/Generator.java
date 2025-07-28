package gan.network;

import gan.core.Tensor;
import gan.layers.*;

public class Generator {
    private final DenseLayer fc1;
    private final ReluLayer relu1;
    private final ReluLayer relu2;
    private final DenseLayer fc2;
    private final TanhLayer tanh;
    private final DenseLayer fc3;

    public Generator(int noiseDim, int outputDim) {
        this.fc1 = new DenseLayer(noiseDim, 256);
        this.relu1 = new ReluLayer();
        this.fc2 = new DenseLayer(256, 512);
        this.relu2 = new ReluLayer();
        this.fc3 = new DenseLayer(512, outputDim);
        this.tanh = new TanhLayer();
    }

    public Tensor forward(Tensor noise) {
        Tensor x = fc1.forward(noise);
        x = relu1.forward(x);
        x = fc2.forward(x);
        x = relu2.forward(x);
        x = fc3.forward(x);
        x = tanh.forward(x);
        return x;
    }

    public Tensor backward(Tensor gradOutput) {
        gradOutput = tanh.backward(gradOutput);
        gradOutput = fc3.backward(gradOutput);
        gradOutput = relu2.backward(gradOutput);
        gradOutput = fc2.backward(gradOutput);
        gradOutput = relu1.backward(gradOutput);
        return fc1.backward(gradOutput);
    }
}
