package alexnet.data;

import alexnet.core.Tensor;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class DataLoader {
    private List<File> imageFiles = new ArrayList<>();
    private List<Integer> labels = new ArrayList<>();
    private Map<String, Integer> labelMap = new HashMap<>();
    private int currentIndex = 0;
    private int imageWidth;
    private int imageHeight;
    private int channels;

    public DataLoader(String datasetPath, int imageWidth, int imageHeight, int channels) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.channels = channels;
        loadDataset(new File(datasetPath));
        shuffleData();
    }

    private void loadDataset(File root) {
        File[] classDirs = root.listFiles(File::isDirectory);
        if (classDirs == null) return;

        int labelIndex = 0;
        for (File classDir : classDirs) {
            String className = classDir.getName();
            File imagesDir = new File(classDir, "images");
            if (!imagesDir.exists() || !imagesDir.isDirectory()) continue;

            File[] imageFilesInDir = imagesDir.listFiles((dir, name) ->
                    name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".jpeg") || name.toLowerCase().endsWith(".png"));

            if (imageFilesInDir == null) continue;

            labelMap.put(className, labelIndex);

            for (File imgFile : imageFilesInDir) {
                imageFiles.add(imgFile);
                labels.add(labelIndex);
            }

            labelIndex++;
        }
    }


    public void shuffleData() {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < imageFiles.size(); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random());

        List<File> shuffledImages = new ArrayList<>();
        List<Integer> shuffledLabels = new ArrayList<>();
        for (int i : indices) {
            shuffledImages.add(imageFiles.get(i));
            shuffledLabels.add(labels.get(i));
        }
        imageFiles = shuffledImages;
        labels = shuffledLabels;
        currentIndex = 0;
    }

    public boolean hasNextBatch(int batchSize) {
        return currentIndex + batchSize <= imageFiles.size();
    }

    public Batch nextBatch(int batchSize) throws IOException {
        List<Tensor> batchImages = new ArrayList<>();
        List<Integer> batchLabels = new ArrayList<>();

        for (int i = 0; i < batchSize; i++) {
            File imgFile = imageFiles.get(currentIndex);
            BufferedImage img = ImageIO.read(imgFile);
            Tensor tensor = preprocess(img);
            batchImages.add(tensor);
            batchLabels.add(labels.get(currentIndex));
            currentIndex++;
        }

        return new Batch(batchImages, batchLabels);
    }

    private Tensor preprocess(BufferedImage img) {
        BufferedImage resized = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(img, 0, 0, imageWidth, imageHeight, null);
        g.dispose();

        float[][][] data = new float[channels][imageHeight][imageWidth];

        for (int y = 0; y < imageHeight; y++) {
            for (int x = 0; x < imageWidth; x++) {
                int rgb = resized.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int gVal = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                data[0][y][x] = r / 255.0f;
                if (channels > 1) {
                    data[1][y][x] = gVal / 255.0f;
                    data[2][y][x] = b / 255.0f;
                }
            }
        }

        return new Tensor(data);
    }

    public static class Batch {
        public final List<Tensor> images;
        public final List<Integer> labels;

        public Batch(List<Tensor> images, List<Integer> labels) {
            this.images = images;
            this.labels = labels;
        }
    }
}
