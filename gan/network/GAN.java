package gan.network;

import gan.core.*;
import gan.loss.BinaryCrossEntropyLoss;
import java.util.Random;

public class GAN {
    private final Generator generator;
    private final Discriminator discriminator;
    private final BinaryCrossEntropyLoss lossFunc;
    private final int noiseDim;

    private final Random rand = new Random();

    public GAN(int noiseDim, int imageDim) {
        this.noiseDim = noiseDim;
        this.generator = new Generator(noiseDim, imageDim);
        this.discriminator = new Discriminator(imageDim);
        this.lossFunc = new BinaryCrossEntropyLoss();
    }

    public Tensor sampleNoise() {
        Tensor noise = new Tensor(1, noiseDim);
        for (int i = 0; i < noiseDim; i++) {
            noise.data[0][i] = rand.nextGaussian();
        }
        return noise;
    }

    public double trainDiscriminator(Tensor realImage) {
        Tensor noise = sampleNoise();
        Tensor fakeImage = generator.forward(noise);

        Tensor realPred = discriminator.forward(realImage);
        Tensor fakePred = discriminator.forward(fakeImage);
        Tensor realLabel = new Tensor(1, 1); realLabel.data[0][0] = 1.0;
        Tensor fakeLabel = new Tensor(1, 1); fakeLabel.data[0][0] = 0.0;

        double lossReal = lossFunc.forward(realPred, realLabel);
        Tensor gradReal = lossFunc.backward();
        discriminator.backward(gradReal);
        double lossFake = lossFunc.forward(fakePred, fakeLabel);
        Tensor gradFake = lossFunc.backward();
        discriminator.backward(gradFake);

        return (lossReal + lossFake) / 2.0;
    }

    public double trainGenerator() {
        Tensor noise = sampleNoise();
        Tensor fakeImage = generator.forward(noise);

        Tensor pred = discriminator.forward(fakeImage);
        Tensor target = new Tensor(1, 1); target.data[0][0] = 1.0;

        double loss = lossFunc.forward(pred, target);
        Tensor grad = lossFunc.backward();

        Tensor gradFromD = discriminator.backward(grad);
        generator.backward(gradFromD);

        return loss;
    }

    public Tensor generate() {
        return generator.forward(sampleNoise());
    }
}
