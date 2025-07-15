package gcn;

import gcn.core.SparseMatrix;
import gcn.data.DatasetLoader;
import gcn.data.GraphPreprocessor;
import gcn.network.GCNModel;

import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {
        // === 路径设置 ===
        String contentPath = "E:/Document/DeepLearning/dataset/cora/cora.content";
        String citesPath = "E:/Document/DeepLearning/dataset/cora/cora.cites";

        // === 加载 .content 文件 ===
        DatasetLoader loader = new DatasetLoader();
        loader.loadContent(contentPath);
        double[][] features = loader.features;
        int[] labels = loader.labels;
        Map<String, Integer> paperIdMap = loader.paperIdMap;
        int numNodes = features.length;

        // === 加载并归一化邻接矩阵 ===
        SparseMatrix normAdj = GraphPreprocessor.buildNormalizedAdj(citesPath, paperIdMap, numNodes);

        // === 数据划分 ===
        int[] trainIndices = new int[2600];
        int[] testIndices = new int[108];
        for (int i = 0; i < 2600; i++) trainIndices[i] = i;
        for (int i = 0; i < 108; i++) testIndices[i] = 2600 + i;

        GCNModel model = new GCNModel(0.01);

        int epochs = 200;
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double[][] predictions = model.forward(normAdj, features);

            int[] trainOnlyLabels = new int[labels.length];
            Arrays.fill(trainOnlyLabels, -1);
            for (int idx : trainIndices) {
                trainOnlyLabels[idx] = labels[idx];
            }

            double loss = model.backward(predictions, trainOnlyLabels);

            model.step();
            if (epoch % 10 == 0 || epoch == 1) {
                double acc = evaluateAccuracy(predictions, labels, testIndices);
                System.out.printf("Epoch %3d | Loss: %.4f | Test Accuracy: %.2f%%%n", epoch, loss, acc * 100);
            }
        }

    }

    private static int argMax(double[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    private static double evaluateAccuracy(double[][] predictions, int[] labels, int[] indices) {
        int correct = 0;
        for (int idx : indices) {
            int predLabel = argMax(predictions[idx]);
            if (predLabel == labels[idx]) {
                correct++;
            }
        }
        return (double) correct / indices.length;
    }

}
