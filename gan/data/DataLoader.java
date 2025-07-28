package gan.data;

import gan.core.Tensor;
import java.io.*;
import java.util.*;

public class DataLoader {
    private final String imagePath;
    private final int imageCount;
    private final int imageSize;  // 28x28 = 784
    private final List<Tensor> images = new ArrayList<>();

    public DataLoader(String imagePath, String labelPath) throws IOException {
        this.imagePath = imagePath;
        this.imageSize = 28 * 28;

        try (DataInputStream imageStream = new DataInputStream(new FileInputStream(imagePath));
             DataInputStream labelStream = new DataInputStream(new FileInputStream(labelPath))) {

            int magicImages = imageStream.readInt();
            int numImages = imageStream.readInt();
            int numRows = imageStream.readInt();
            int numCols = imageStream.readInt();

            int magicLabels = labelStream.readInt();
            int numLabels = labelStream.readInt();

            if (numImages != numLabels)
                throw new IOException("图像数量与标签数量不匹配");

            this.imageCount = numImages;

            for (int i = 0; i < numImages; i++) {
                byte[] imgBytes = new byte[imageSize];
                imageStream.readFully(imgBytes);
                Tensor img = ImagePreprocessor.preprocessImage(imgBytes);
                images.add(img);

                // 标签可以忽略，读掉以防数据错位
                labelStream.readByte();
            }
        }
    }

    public int size() {
        return imageCount;
    }

    public List<Tensor> getImages() {
        return images;
    }
}
