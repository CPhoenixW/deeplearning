package gcn.optim;

public class SGD implements Optimizer {
    private final double learningRate;

    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void update(double[][] weights, double[][] gradWeights) {
        int rows = weights.length;
        int cols = weights[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] -= learningRate * gradWeights[i][j];
            }
        }
    }
}
