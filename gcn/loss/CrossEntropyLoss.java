package gcn.loss;

public class CrossEntropyLoss {

    public double forward(double[][] predictions, int[] targets) {
        double loss = 0.0;
        int count = 0;
        for (int i = 0; i < predictions.length; i++) {
            int label = targets[i];
            if (label == -1) continue;
            loss -= Math.log(predictions[i][label]);
            count++;
        }
        return loss / count;
    }

    public double[][] backward(double[][] predictions, int[] targets) {
        double[][] grad = new double[predictions.length][predictions[0].length];
        int count = 0;
        for (int i = 0; i < predictions.length; i++) {
            int label = targets[i];
            if (label == -1) continue;
            for (int j = 0; j < predictions[i].length; j++) {
                grad[i][j] = predictions[i][j];
            }
            grad[i][label] -= 1;
            count++;
        }
        for (int i = 0; i < predictions.length; i++) {
            for (int j = 0; j < predictions[i].length; j++) {
                grad[i][j] /= count;
            }
        }
        return grad;
    }
}
