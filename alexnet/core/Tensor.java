package alexnet.core;

import java.util.Arrays;

public class Tensor {
    private final float[][][] data;
    private final int channels;
    private final int height;
    private final int width;

    public Tensor(int channels, int height, int width) {
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.data = new float[channels][height][width];
    }

    public Tensor(float[][][] data) {
        this.channels = data.length;
        this.height = data[0].length;
        this.width = data[0][0].length;
        this.data = new float[channels][height][width];
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                System.arraycopy(data[c][h], 0, this.data[c][h], 0, width);
            }
        }
    }

    public int getChannels() {
        return channels;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public float get(int c, int h, int w) {
        return data[c][h][w];
    }

    public void set(int c, int h, int w, float value) {
        data[c][h][w] = value;
    }

    public float[][][] getData() {
        return data;
    }

    public Tensor copy() {
        return new Tensor(this.data);
    }

    public void fill(float value) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                Arrays.fill(data[c][h], value);
            }
        }
    }

    public void addInPlace(Tensor other) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    this.data[c][h][w] += other.data[c][h][w];
                }
            }
        }
    }

    public Tensor apply(ReLUFunction func) {
        Tensor out = new Tensor(channels, height, width);
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    out.data[c][h][w] = func.apply(data[c][h][w]);
                }
            }
        }
        return out;
    }

    public interface ReLUFunction {
        float apply(float x);
    }

    @Override
    public String toString() {
        return "Tensor{" +
                "channels=" + channels +
                ", height=" + height +
                ", width=" + width +
                '}';
    }

    public static boolean shapeEquals(Tensor a, Tensor b) {
        float[][][] d1 = a.getData();
        float[][][] d2 = b.getData();
        return d1.length == d2.length &&
                d1[0].length == d2[0].length &&
                d1[0][0].length == d2[0][0].length;
    }

    public String getShapeString() {
        return "[" + getChannels() + "][" + getHeight() + "][" + getWidth() + "]";
    }

}
