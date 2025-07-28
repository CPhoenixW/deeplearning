package gan.utils;

import gan.core.Tensor;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Visualizer {

    // 保存 1x784 Tensor 到 PNG 文件，值假设 [-1, 1]
    public static void saveImage(Tensor tensor, String filepath) throws IOException {
        int size = (int) Math.sqrt(tensor.cols);
        BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_BYTE_GRAY);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double val = tensor.data[0][i * size + j];
                int gray = (int) ((val + 1) / 2 * 255);
                gray = Math.min(255, Math.max(0, gray));
                int rgb = (gray << 16) | (gray << 8) | gray;
                img.setRGB(j, i, rgb);
            }
        }
        ImageIO.write(img, "png", new File(filepath));
    }

    // 用 Swing 显示图片（阻塞窗口）
    public static void showImage(Tensor tensor, String title) {
        int size = (int) Math.sqrt(tensor.cols);
        BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_BYTE_GRAY);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double val = tensor.data[0][i * size + j];
                int gray = (int) ((val + 1) / 2 * 255);
                gray = Math.min(255, Math.max(0, gray));
                int rgb = (gray << 16) | (gray << 8) | gray;
                img.setRGB(j, i, rgb);
            }
        }

        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(size * 10, size * 10);
        JLabel label = new JLabel(new ImageIcon(img.getScaledInstance(size * 10, size * 10, Image.SCALE_FAST)));
        frame.add(label);
        frame.pack();
        frame.setVisible(true);
    }
}
