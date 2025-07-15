package gcn.core;

import java.util.Random;


public class Initializer {

    private static final Random random = new Random();

    public static double[][] xavier(int fanIn, int fanOut) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        double[][] weights = new double[fanIn][fanOut];
        for (int i = 0; i < fanIn; i++) {
            for (int j = 0; j < fanOut; j++) {
                weights[i][j] = random.nextDouble() * 2 * limit - limit;
            }
        }
        return weights;
    }
}
