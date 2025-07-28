package gan.core;

public class MatrixUtils {

    public static Tensor dot(Tensor a, Tensor b) {
        if (a.cols != b.rows)
            throw new IllegalArgumentException("Incompatible shapes for matrix multiplication");

        Tensor result = new Tensor(a.rows, b.cols);

        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < b.cols; j++) {
                for (int k = 0; k < a.cols; k++) {
                    result.data[i][j] += a.data[i][k] * b.data[k][j];
                }
            }
        }

        return result;
    }

    public static Tensor add(Tensor a, Tensor b) {
        Tensor result = new Tensor(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }
        return result;
    }

    public static Tensor subtract(Tensor a, Tensor b) {
        Tensor result = new Tensor(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }

    public static Tensor multiplyElementwise(Tensor a, Tensor b) {
        Tensor result = new Tensor(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] * b.data[i][j];
            }
        }
        return result;
    }

    public static Tensor applyFunction(Tensor input, java.util.function.Function<Double, Double> func) {
        Tensor result = new Tensor(input.rows, input.cols);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                result.data[i][j] = func.apply(input.data[i][j]);
            }
        }
        return result;
    }

    public static Tensor transpose(Tensor input) {
        Tensor result = new Tensor(input.cols, input.rows);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                result.data[j][i] = input.data[i][j];
            }
        }
        return result;
    }

    public static Tensor scalarMultiply(Tensor a, double scalar) {
        Tensor result = new Tensor(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] * scalar;
            }
        }
        return result;
    }
}
