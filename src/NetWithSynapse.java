import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import sun.plugin.javascript.navig4.Layer;

/**
 * Created by Teng on 01/07/2014.
 */
public class NetWithSynapse {
    public static void main(String[] args){
        BasicNetwork network = new BasicNetwork();

        BasicLayer inputLayer = new BasicLayer(new ActivationSigmoid(), true, 2);
        BasicLayer hiddenLayer = new BasicLayer(new ActivationSigmoid(), true, 2);
        BasicLayer outputLayer = new BasicLayer(new ActivationSigmoid(), true, 1);

        NetWithSynapse synapseInputToHidden = new NetWithSynapse();

    }
}
