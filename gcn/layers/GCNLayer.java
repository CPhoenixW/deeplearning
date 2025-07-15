package gcn.layers;

import gcn.core.Initializer;
import gcn.core.MatrixUtils;
import gcn.core.SparseMatrix;
import gcn.optim.Optimizer;


public class GCNLayer {
    private final int inputDim;
    private final int outputDim;
    private final double[][] weights;
    private final double[][] gradWeights;
    private final Optimizer optimizer;

    private double[][] inputX;
    private SparseMatrix normAdj;

    public GCNLayer(int inputDim, int outputDim, Optimizer optimizer) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.weights = Initializer.xavier(inputDim, outputDim);
        this.gradWeights = new double[inputDim][outputDim];
        this.optimizer = optimizer;
    }

    public double[][] forward(SparseMatrix adj, double[][] inputFeatures) {
        this.inputX = inputFeatures;
        this.normAdj = adj;
        return normAdj.dot(MatrixUtils.dot(inputX, weights));
    }

    public double[][] backward(double[][] gradOutput) {
        double[][] temp = MatrixUtils.dot(MatrixUtils.transpose(inputX), normAdj.transpose().dot(gradOutput));
        copy(temp, gradWeights);
        return normAdj.dot(MatrixUtils.dot(gradOutput, MatrixUtils.transpose(weights)));
    }

    public void step() {
        optimizer.update(weights, gradWeights);
    }

    private static void copy(double[][] src, double[][] dest) {
        for (int i = 0; i < src.length; i++)
            System.arraycopy(src[i], 0, dest[i], 0, src[0].length);
    }
}
