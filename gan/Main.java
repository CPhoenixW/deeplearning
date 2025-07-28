package gan;

import gan.core.Tensor;
import gan.data.DataLoader;
import gan.network.GAN;
import gan.utils.Logger;
import gan.utils.Visualizer;
import java.io.IOException;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        // 路径配置
        String trainImages = "E:/Document/DeepLearning/dataset/FashionMNIST/raw/train-images-idx3-ubyte";
        String trainLabels = "E:/Document/DeepLearning/dataset/FashionMNIST/raw/train-labels-idx1-ubyte";

        // 加载数据
        DataLoader dataLoader = new DataLoader(trainImages, trainLabels);
        // 模型参数
        int noiseDim = 100;
        int imageDim = 28 * 28;

        GAN gan = new GAN(noiseDim, imageDim);

        int epochs = 50;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            System.out.println("Epoch " + epoch);
            double totalLossD = 0;
            double totalLossG = 0;
            int sampleCount = 0;

            List<Tensor> images = dataLoader.getImages();

            for (Tensor realImg : images) {
                totalLossD += gan.trainDiscriminator(realImg);
                totalLossG += gan.trainGenerator();
                sampleCount++;
            }

            double avgLossD = totalLossD / sampleCount;
            double avgLossG = totalLossG / sampleCount;

            Logger.logEpochLoss(epoch, avgLossD, avgLossG);

            if (epoch % 10 == 0) {
                Tensor sample = gan.generate();
                String filename = String.format("generated_epoch_%d.png", epoch);
                Visualizer.saveImage(sample, filename);
                Logger.log("MAIN", "Saved generated image: " + filename);
            }
        }

        Logger.log("MAIN", "Training complete.");
    }
}
