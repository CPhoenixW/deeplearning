package alexnet;

import alexnet.core.Tensor;
import alexnet.data.DataLoader;
import alexnet.data.DataLoader.Batch;
import alexnet.loss.CrossEntropyLoss;
import alexnet.network.AlexNet;

import java.io.IOException;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        int imageWidth = 227;
        int imageHeight = 227;
        int channels = 3;
        int batchSize = 8;
        int numEpochs = 5;
        double learningRate = 0.01;
        int numClasses = 1000; // 请根据数据集实际类数修改！

        try {
            DataLoader loader = new DataLoader("E:\\Document\\DeepLearning\\dataset\\tiny-imagenet-200", imageWidth, imageHeight, channels);
            AlexNet net = new AlexNet();
            CrossEntropyLoss lossFn = new CrossEntropyLoss();

            for (int epoch = 1; epoch <= numEpochs; epoch++) {
                System.out.println("Epoch " + epoch + " starting...");
                loader.shuffleData();

                int batchCount = 0;
                float totalLoss = 0;
                int totalCorrect = 0;
                int totalSamples = 0;

                while (loader.hasNextBatch(batchSize)) {
                    Batch batch = loader.nextBatch(batchSize);
                    List<Tensor> images = batch.images;
                    List<Integer> labels = batch.labels;

                    for (int i = 0; i < images.size(); i++) {
                        Tensor input = images.get(i);
                        int trueLabel = labels.get(i);

                        // One-hot 标签
                        float[][][] labelData = new float[1][1][numClasses];
                        labelData[0][0][trueLabel] = 1.0f;
                        Tensor target = new Tensor(labelData);

                        // Forward → loss
                        Tensor output = net.forward(input);
                        float loss = lossFn.forward(output, target);
                        totalLoss += loss;

                        // Accuracy
                        int predicted = argMax(output);
                        if (predicted == trueLabel) {
                            totalCorrect++;
                        }

                        // Backward
                        Tensor gradLoss = lossFn.backward(output, target);
                        net.backward(gradLoss, learningRate);

                        totalSamples++;
                    }

                    batchCount++;
                    System.out.println("Batch " + batchCount + " processed");
                }

                float avgLoss = totalLoss / totalSamples;
                float accuracy = (float) totalCorrect / totalSamples * 100;
                System.out.printf("Epoch %d completed. Avg Loss: %.4f | Accuracy: %.2f%%\n", epoch, avgLoss, accuracy);
            }

        } catch (IOException e) {
            System.err.println("Failed to load dataset: " + e.getMessage());
        }
    }

    private static int argMax(Tensor output) {
        float[][][] data = output.getData();
        float[] flat = data[0][0];
        int maxIdx = 0;
        float maxVal = flat[0];
        for (int i = 1; i < flat.length; i++) {
            if (flat[i] > maxVal) {
                maxVal = flat[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
