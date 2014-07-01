import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.data.NeuralData;
import org.encog.neural.data.NeuralDataPair;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import sun.applet.Main;

/**
 * Created by Teng on 01/07/2014.
 */
public class Test {
    public static double XOR_INPUT[][] = {
            {0.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 1.0}
    };

    public static double XOR_IDEAL[][] = {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
    };

    public static void main(String[] args){
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 2));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 4));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));

        network.getStructure().finalizeStructure();
        network.reset();

        MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

//        Train the net
        final Train train = new Backpropagation(network, trainingSet);

        int epoch = 1;

        do {
            train.iteration();
            System.out.println("Epoch:" + epoch+ " Error:" + train.getError() );
            epoch++;
        }while (train.getError() > 0.01);

//        Test Neural Net
        System.out.println("Neural network result:");
        for (MLDataPair pair: trainingSet){
            final MLData output = network.compute((MLData)pair.getInput());
            System.out.println(pair.getInput().getData(0)+","+pair.getInput().getData(1) + ", actual=" + output.getData(0)+", ideal=" + pair.getIdeal().getData(0));

        }
    }
}
