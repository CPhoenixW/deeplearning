package gan.core;

import java.util.Arrays;

public class Tensor {
    public double[][] data;
    public int rows;
    public int cols;

    public Tensor(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Tensor(double[][] data) {
        this.data = data;
        this.rows = data.length;
        this.cols = data[0].length;
    }

    public static Tensor fromFlatArray(double[] arr) {
        Tensor t = new Tensor(arr.length, 1);
        for (int i = 0; i < arr.length; i++) {
            t.data[i][0] = arr[i];
        }
        return t;
    }

    public Tensor copy() {
        double[][] newData = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            newData[i] = Arrays.copyOf(data[i], cols);
        }
        return new Tensor(newData);
    }

    public void print() {
        for (double[] row : data) {
            System.out.println(Arrays.toString(row));
        }
    }
}
