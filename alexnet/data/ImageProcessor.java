package alexnet.data;

import alexnet.core.Tensor;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageProcessor {

    private int targetWidth;
    private int targetHeight;

    // 以AlexNet为例，常用均值和标准差 (BGR顺序)
    private final float[] mean = {0.485f, 0.456f, 0.406f};
    private final float[] std = {0.229f, 0.224f, 0.225f};

    public ImageProcessor(int targetWidth, int targetHeight) {
        this.targetWidth = targetWidth;
        this.targetHeight = targetHeight;
    }

    /**
     * 读取图片文件，调整大小，转为归一化Tensor（float, RGB）
     * @param path 图片文件路径
     * @return Tensor shape: [3][targetHeight][targetWidth]
     * @throws IOException
     */
    public Tensor loadImageAsTensor(String path) throws IOException {
        BufferedImage img = ImageIO.read(new File(path));
        BufferedImage resized = resizeImage(img, targetWidth, targetHeight);
        return bufferedImageToTensor(resized);
    }

    private BufferedImage resizeImage(BufferedImage originalImage, int width, int height) {
        Image tmp = originalImage.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        return resized;
    }

    private Tensor bufferedImageToTensor(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();

        float[][][] data = new float[3][h][w];  // RGB 顺序

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                // 归一化到 [0,1]
                float fr = r / 255.0f;
                float fg = g / 255.0f;
                float fb = b / 255.0f;

                // 标准化: (x - mean) / std
                data[0][y][x] = (fr - mean[0]) / std[0];
                data[1][y][x] = (fg - mean[1]) / std[1];
                data[2][y][x] = (fb - mean[2]) / std[2];
            }
        }

        return new Tensor(data);
    }
}
