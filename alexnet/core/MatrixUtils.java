package alexnet.core;

public class MatrixUtils {

    /**
     * 矩阵乘法： A[m×n] * B[n×p] = C[m×p]
     * @param A 左矩阵
     * @param B 右矩阵
     * @return 乘积矩阵
     */
    public static float[][] multiply(float[][] A, float[][] B) {
        int m = A.length;
        int n = A[0].length;
        int p = B[0].length;

        if (B.length != n) {
            throw new IllegalArgumentException("矩阵维度不匹配");
        }

        float[][] C = new float[m][p];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                float sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    }

    /**
     * 矩阵转置： M[m×n] -> M^T[n×m]
     * @param M 输入矩阵
     * @return 转置矩阵
     */
    public static float[][] transpose(float[][] M) {
        int m = M.length;
        int n = M[0].length;
        float[][] T = new float[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                T[j][i] = M[i][j];
            }
        }
        return T;
    }

    /**
     * 矩阵加法： A + B
     * @param A 矩阵A
     * @param B 矩阵B
     * @return 结果矩阵
     */
    public static float[][] add(float[][] A, float[][] B) {
        int m = A.length;
        int n = A[0].length;

        if (B.length != m || B[0].length != n) {
            throw new IllegalArgumentException("矩阵维度不匹配");
        }

        float[][] C = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
        return C;
    }

    /**
     * 矩阵元素逐个相乘（Hadamard积）
     * @param A 矩阵A
     * @param B 矩阵B
     * @return 结果矩阵
     */
    public static float[][] elementWiseMultiply(float[][] A, float[][] B) {
        int m = A.length;
        int n = A[0].length;

        if (B.length != m || B[0].length != n) {
            throw new IllegalArgumentException("矩阵维度不匹配");
        }

        float[][] C = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] * B[i][j];
            }
        }
        return C;
    }

    /**
     * 生成零矩阵
     * @param rows 行数
     * @param cols 列数
     * @return 零矩阵
     */
    public static float[][] zeros(int rows, int cols) {
        return new float[rows][cols];
    }
}
