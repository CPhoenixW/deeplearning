package gcn.optim;

public class Adam implements Optimizer {
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private double[][] m;
    private double[][] v;
    private int t = 0;
    private boolean initialized = false;

    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    private void initialize(int rows, int cols) {
        this.m = new double[rows][cols];
        this.v = new double[rows][cols];
        this.initialized = true;
    }

    @Override
    public void update(double[][] weights, double[][] gradWeights) {
        int rows = weights.length;
        int cols = weights[0].length;
        if (!initialized) {
            initialize(rows, cols);
        }

        t += 1;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * gradWeights[i][j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * gradWeights[i][j] * gradWeights[i][j];

                double mHat = m[i][j] / (1 - Math.pow(beta1, t));
                double vHat = v[i][j] / (1 - Math.pow(beta2, t));

                weights[i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    }
}
