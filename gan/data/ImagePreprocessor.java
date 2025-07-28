package gan.data;

import gan.core.Tensor;

public class ImagePreprocessor {

    // 将28x28图像扁平化成 (1, 784) 并归一化到 0~1
    public static Tensor preprocessImage(byte[] image) {
        int size = image.length;
        Tensor tensor = new Tensor(1, size);
        for (int i = 0; i < size; i++) {
            int pixel = image[i] & 0xFF;
            tensor.data[0][i] = pixel / 255.0;
        }
        return tensor;
    }

    // 标签转为 one-hot 向量 (可选)
    public static Tensor oneHotEncode(byte label, int numClasses) {
        Tensor t = new Tensor(1, numClasses);
        t.data[0][label] = 1.0;
        return t;
    }
}
