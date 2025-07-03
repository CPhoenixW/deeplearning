package alexnet.layers;

import alexnet.core.Tensor;
import alexnet.network.Layer;

public class FlattenLayer implements Layer {
    private Tensor inputCache;

    @Override
    public Tensor forward(Tensor input) {
        this.inputCache = input;

        int c = input.getChannels();
        int h = input.getHeight();
        int w = input.getWidth();
        float[][][] data = input.getData();

        float[] flat = new float[c * h * w];
        int idx = 0;
        for (int ch = 0; ch < c; ch++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    flat[idx++] = data[ch][i][j];
                }
            }
        }

        float[][][] outData = new float[1][1][flat.length];
        outData[0][0] = flat;

        return new Tensor(outData);
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        int c = inputCache.getChannels();
        int h = inputCache.getHeight();
        int w = inputCache.getWidth();

        float[] flatGrad = gradOutput.getData()[0][0];
        float[][][] gradInput = new float[c][h][w];

        int idx = 0;
        for (int ch = 0; ch < c; ch++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    gradInput[ch][i][j] = flatGrad[idx++];
                }
            }
        }

        return new Tensor(gradInput);
    }
}
