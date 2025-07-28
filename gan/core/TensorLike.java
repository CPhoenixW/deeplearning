package gan.core;

public class TensorLike extends Tensor {
    public TensorLike(double value) {
        super(1, 1);
        this.data[0][0] = value;
    }
}

