package gcn.core;

import java.util.*;

/**
 * 简单实现：基于邻接表的稀疏矩阵
 * 主要用于图中的邻接矩阵存储和操作
 */
public class SparseMatrix {

    private final int rows;
    private final int cols;
    private final Map<Integer, Map<Integer, Double>> data;

    public SparseMatrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new HashMap<>();
    }

    public void set(int row, int col, double value) {
        if (!data.containsKey(row)) {
            data.put(row, new HashMap<>());
        }
        data.get(row).put(col, value);
    }

    public double get(int row, int col) {
        return data.getOrDefault(row, Collections.emptyMap()).getOrDefault(col, 0.0);
    }

    public void addEdge(int from, int to) {
        set(from, to, 1.0);
    }

    public void addSelfLoops() {
        for (int i = 0; i < rows; i++) {
            set(i, i, 1.0);
        }
    }

    public double[] computeDegreeVector() {
        double[] degree = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0;
            Map<Integer, Double> row = data.getOrDefault(i, Collections.emptyMap());
            for (double v : row.values()) {
                sum += v;
            }
            degree[i] = sum;
        }
        return degree;
    }

    public SparseMatrix normalizeSymmetric() {
        this.addSelfLoops();
        double[] degree = computeDegreeVector();
        SparseMatrix norm = new SparseMatrix(rows, cols);

        for (int i : data.keySet()) {
            for (int j : data.get(i).keySet()) {
                double v = data.get(i).get(j);
                double d_i = degree[i] == 0 ? 0 : 1.0 / Math.sqrt(degree[i]);
                double d_j = degree[j] == 0 ? 0 : 1.0 / Math.sqrt(degree[j]);
                norm.set(i, j, v * d_i * d_j);
            }
        }

        return norm;
    }

    public double[][] dot(double[][] dense) {
        int outDim = dense[0].length;
        double[][] result = new double[rows][outDim];

        for (int i : data.keySet()) {
            Map<Integer, Double> row = data.get(i);
            for (int j : row.keySet()) {
                double val = row.get(j);
                for (int k = 0; k < outDim; k++) {
                    result[i][k] += val * dense[j][k];
                }
            }
        }

        return result;
    }

    public SparseMatrix transpose() {
        SparseMatrix transposed = new SparseMatrix(cols, rows);

        for (int i : data.keySet()) {
            for (int j : data.get(i).keySet()) {
                double value = data.get(i).get(j);
                transposed.set(j, i, value);
            }
        }

        return transposed;
    }

}
