import java.io.File;

import org.encog.Encog;
import org.encog.mathutil.randomize.ConsistentRandomizer;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.simple.EncogUtility;

/**
 * Created by Teng on 05/07/2014.
 */

/**
 * This example shows how to use Encog persistence to store a neural network
 * to an EG file.  The EG file is cross-platform and can be shared between
 * Encog Java and Encog C#.
 *
 */
public class EncogPersistence {

    public static final String FILENAME = "encogexample.eg";

    public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
            { 0.0, 1.0 }, { 1.0, 1.0 } };

    public static double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };

    public void trainAndSave() {
        System.out.println("Training XOR network to under 1% error rate.");
        BasicNetwork network = EncogUtility.simpleFeedForward(2, 3, 0, 1, false);

        // randomize consistent so that we get weights we know will converge
        (new ConsistentRandomizer(-1,1,100)).randomize(network);

        MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

        // train the neural network
        final MLTrain train = new ResilientPropagation(network, trainingSet);

        do {
            train.iteration();
        } while (train.getError() > 0.009);

        double e = network.calculateError(trainingSet);
        System.out.println("Network traiined to error: " + e);

        System.out.println("Saving network");
        EncogDirectoryPersistence.saveObject(new File(FILENAME), network);
    }

    public void loadAndEvaluate() {
        System.out.println("Loading network");

        BasicNetwork network = (BasicNetwork)EncogDirectoryPersistence.loadObject(new File(FILENAME));

        MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
        double e = network.calculateError(trainingSet);
        System.out
                .println("Loaded network's error is(should be same as above): "
                        + e);
    }

    public static void main(String[] args) {
        try {
            EncogPersistence program = new EncogPersistence();
            program.trainAndSave();
            program.loadAndEvaluate();
        } catch (Throwable t) {
            t.printStackTrace();
        } finally {
            Encog.getInstance().shutdown();
        }

    }
}

