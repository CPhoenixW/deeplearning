package gcn.layers;

public class ReluLayer {
    private double[][] input;

    public double[][] forward(double[][] input) {
        this.input = input;
        int rows = input.length;
        int cols = input[0].length;
        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                output[i][j] = Math.max(0, input[i][j]);

        return output;
    }

    public double[][] backward(double[][] gradOutput) {
        int rows = input.length;
        int cols = input[0].length;
        double[][] gradInput = new double[rows][cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                gradInput[i][j] = input[i][j] > 0 ? gradOutput[i][j] : 0.0;

        return gradInput;
    }
}
