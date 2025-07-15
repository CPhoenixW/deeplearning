package gcn.layers;

public class SoftmaxLayer {
    private double[][] output;

    public double[][] forward(double[][] input) {
        int rows = input.length;
        int cols = input[0].length;
        output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) {
                if (input[i][j] > max) max = input[i][j];
            }

            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                output[i][j] = Math.exp(input[i][j] - max);
                sum += output[i][j];
            }

            for (int j = 0; j < cols; j++) {
                output[i][j] /= sum;
            }
        }

        return output;
    }

    public double[][] backward(double[][] gradOutput) {
        return gradOutput;
    }

    public double[][] getOutput() {
        return output;
    }
}
