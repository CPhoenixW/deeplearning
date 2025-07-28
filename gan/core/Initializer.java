package gan.core;

import java.util.Random;

public class Initializer {

    private static final Random rand = new Random();

    public static Tensor randomNormal(int rows, int cols, double mean, double stdDev) {
        Tensor t = new Tensor(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t.data[i][j] = mean + stdDev * rand.nextGaussian();
            }
        }
        return t;
    }

    public static Tensor zeros(int rows, int cols) {
        return new Tensor(rows, cols);
    }

    public static Tensor xavier(int inDim, int outDim) {
        double stdDev = Math.sqrt(2.0 / (inDim + outDim));
        return randomNormal(outDim, inDim, 0.0, stdDev);
    }

    public static Tensor he(int inDim, int outDim) {
        double stdDev = Math.sqrt(2.0 / inDim);
        return randomNormal(outDim, inDim, 0.0, stdDev);
    }
}
