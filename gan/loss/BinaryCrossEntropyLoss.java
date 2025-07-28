package gan.loss;

import gan.core.*;

public class BinaryCrossEntropyLoss {
    private Tensor predictions;
    private Tensor targets;
    private double epsilon = 1e-12;

    public double forward(Tensor predictions, Tensor targets) {
        this.predictions = predictions;
        this.targets = targets;

        double loss = 0.0;
        for (int i = 0; i < predictions.rows; i++) {
            for (int j = 0; j < predictions.cols; j++) {
                double y = targets.data[i][j];
                double p = clamp(predictions.data[i][j]);
                loss += -y * Math.log(p) - (1 - y) * Math.log(1 - p);
            }
        }

        return loss / (predictions.rows * predictions.cols);
    }

    public Tensor backward() {
        Tensor grad = new Tensor(predictions.rows, predictions.cols);

        for (int i = 0; i < predictions.rows; i++) {
            for (int j = 0; j < predictions.cols; j++) {
                double y = targets.data[i][j];
                double p = clamp(predictions.data[i][j]);

                grad.data[i][j] = -(y / p) + (1 - y) / (1 - p);
            }
        }

        int total = predictions.rows * predictions.cols;
        for (int i = 0; i < grad.rows; i++) {
            for (int j = 0; j < grad.cols; j++) {
                grad.data[i][j] /= total;
            }
        }

        return grad;
    }

    private double clamp(double value) {
        return Math.max(epsilon, Math.min(1.0 - epsilon, value));
    }
}
