package gcn.network;

import gcn.core.SparseMatrix;
import gcn.layers.GCNLayer;
import gcn.layers.ReluLayer;
import gcn.layers.SoftmaxLayer;
import gcn.loss.CrossEntropyLoss;
import gcn.optim.Adam;
import gcn.optim.Optimizer;

/**
 * 两层 GCN 模型：GCN + ReLU + GCN + Softmax
 */
public class GCNModel {
    private final GCNLayer layer1;
    private final GCNLayer layer2;
    private final ReluLayer relu;
    private final SoftmaxLayer softmax;
    private final CrossEntropyLoss lossFn;

    private double[][] out1;
    private double[][] out2;

    public GCNModel(double learningRate) {
        Optimizer opt1 = new Adam(learningRate);
        Optimizer opt2 = new Adam(learningRate);
        this.layer1 = new GCNLayer(1433, 16, opt1);
        this.layer2 = new GCNLayer(16, 7, opt2);
        this.relu = new ReluLayer();
        this.softmax = new SoftmaxLayer();
        this.lossFn = new CrossEntropyLoss();
    }

    /**
     * 前向传播：Z = GCN1 → ReLU → GCN2 → Softmax
     */
    public double[][] forward(SparseMatrix adj, double[][] features) {
        out1 = layer1.forward(adj, features);
        double[][] reluOut = relu.forward(out1);
        out2 = layer2.forward(adj, reluOut);
        return softmax.forward(out2);
    }

    public double backward(double[][] predictions, int[] labels) {
        double loss = lossFn.forward(predictions, labels);
        double[][] dLoss = lossFn.backward(predictions, labels);

        double[][] dOut2 = layer2.backward(dLoss);
        double[][] dRelu = relu.backward(dOut2);
        layer1.backward(dRelu);

        return loss;
    }

    public void step() {
        layer1.step();
        layer2.step();
    }
}
