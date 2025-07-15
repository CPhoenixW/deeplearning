package gcn.optim;

public interface Optimizer {
    void update(double[][] weights, double[][] gradWeights);
}
