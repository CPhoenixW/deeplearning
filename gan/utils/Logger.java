package gan.utils;

import java.text.SimpleDateFormat;
import java.util.Date;

public class Logger {
    private static final SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");

    public static void log(String tag, String message) {
        System.out.println("[" + sdf.format(new Date()) + "] [" + tag + "] " + message);
    }

    public static void logEpochLoss(int epoch, double lossD, double lossG) {
        log("TRAIN", String.format("Epoch %d - Discriminator Loss: %.4f, Generator Loss: %.4f", epoch, lossD, lossG));
    }
}
