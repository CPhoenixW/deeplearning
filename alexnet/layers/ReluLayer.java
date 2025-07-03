package alexnet.layers;

import alexnet.core.Tensor;
import alexnet.network.Layer;

public class ReluLayer implements Layer {
    private Tensor input;

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        return input.apply(x -> Math.max(0, x));
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
//        if (!Tensor.shapeEquals(input, gradOutput)) {
//            throw new IllegalArgumentException("Shape mismatch in ReLU backward: input="
//                    + input.getShapeString() + ", gradOutput=" + gradOutput.getShapeString());
//        }
        float[][][] gradIn = new float[input.getChannels()][input.getHeight()][input.getWidth()];
        float[][][] inData = input.getData();
        float[][][] gradOutData = gradOutput.getData();

        for (int c = 0; c < input.getChannels(); c++) {
            for (int h = 0; h < input.getHeight(); h++) {
                for (int w = 0; w < input.getWidth(); w++) {
                    gradIn[c][h][w] = inData[c][h][w] > 0 ? gradOutData[c][h][w] : 0;
                }
            }
        }

        return new Tensor(gradIn);
    }
}
