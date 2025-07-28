package gan.network;

import gan.core.Tensor;
import gan.layers.*;

public class Discriminator {
    private final DenseLayer fc1;
    private final LeakyReluLayer relu1;
    private final LeakyReluLayer relu2;
    private final DenseLayer fc2;
    private final DenseLayer fc3;
    private final SigmoidLayer sigmoid;

    public Discriminator(int inputDim) {
        this.fc1 = new DenseLayer(inputDim, 512);
        this.relu1 = new LeakyReluLayer();
        this.fc2 = new DenseLayer(512, 256);
        this.relu2 = new LeakyReluLayer();
        this.fc3 = new DenseLayer(256, 1);
        this.sigmoid = new SigmoidLayer();
    }

    public Tensor forward(Tensor input) {
        Tensor x = fc1.forward(input);
        x = relu1.forward(x);
        x = fc2.forward(x);
        x = relu2.forward(x);
        x = fc3.forward(x);
        x = sigmoid.forward(x);
        return x;
    }

    public Tensor backward(Tensor gradOutput) {
        gradOutput = sigmoid.backward(gradOutput);
        gradOutput = fc3.backward(gradOutput);
        gradOutput = relu2.backward(gradOutput);
        gradOutput = fc2.backward(gradOutput);
        gradOutput = relu1.backward(gradOutput);
        return fc1.backward(gradOutput);
    }
}
