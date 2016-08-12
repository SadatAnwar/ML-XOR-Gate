import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class XorGate
{
    private static final int OUT_NEURONS = 1; //TODO: Switch between 0 and 1 to change the neurons in the OP layer

    private static final int HIDDEN_NEURONS = 2;

    private static final int IN_NEURONS = 2;

    public static void main(String... args)
    {
        new XorGate();
    }

    public XorGate()
    {
        MultiLayerNetwork nn = getNeuralNetwork();
        nn.init();
        nn.setListeners(new ScoreIterationListener(100));
        DataSet trainingData;
        if (OUT_NEURONS == 2) {
            trainingData = getTrainingDataDoubleOut();
        } else {
            trainingData = getTrainingDataSingleOut();
        }

        DataSetIterator iterator = new TestDataSetIterator(trainingData);

        for (int i = 0; i < 500; i++) {
            iterator.reset();
            nn.fit(iterator);
        }

        //Test the network here

        INDArray testInputs = Nd4j.create(new double[]{1, 1});
        System.out.println(nn.output(testInputs).toString());

        testInputs = Nd4j.create(new double[]{0, 0});
        System.out.println(nn.output(testInputs).toString());

        testInputs = Nd4j.create(new double[]{0, 1});
        System.out.println(nn.output(testInputs).toString());

        testInputs = Nd4j.create(new double[]{1, 0});
        System.out.println(nn.output(testInputs).toString());
    }

    private MultiLayerNetwork getNeuralNetwork()
    {
        return new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(123)
            .iterations(100)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(0.1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(IN_NEURONS).nOut(HIDDEN_NEURONS)
                .activation("tanh")
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation("sigmoid")
                .nIn(HIDDEN_NEURONS).nOut(OUT_NEURONS).build())
            .pretrain(false)
            .backprop(true)
            .build()
        );
    }

    /**
     * Generate a training dataset for the XOR function to train on
     * <p>
     * The data set should have inputs with dimension 4X2 and labels 4X2 (2 columns in labels as its a classification)
     *
     * @return a training dataSet
     */
    private DataSet getTrainingDataSingleOut()
    {
        INDArray input = Nd4j.zeros(4, 2);
        INDArray outPut = Nd4j.zeros(4, 1);
        input.putRow(0, Nd4j.create(new double[]{0, 0}));
        outPut.putRow(0, Nd4j.create(new double[]{0}));

        input.putRow(1, Nd4j.create(new double[]{1, 1}));
        outPut.putRow(1, Nd4j.create(new double[]{0}));

        input.putRow(2, Nd4j.create(new double[]{1, 0}));
        outPut.putRow(2, Nd4j.create(new double[]{1}));

        input.putRow(3, Nd4j.create(new double[]{0, 1}));
        outPut.putRow(3, Nd4j.create(new double[]{1}));

        System.out.println(input);
        System.out.println(outPut);
        return new DataSet(input, outPut);
    }

    private DataSet getTrainingDataDoubleOut()
    {
        INDArray input = Nd4j.zeros(4, 2);
        INDArray outPut = Nd4j.zeros(4, 2);
        input.putRow(0, Nd4j.create(new double[]{0, 0}));
        outPut.putRow(0, Nd4j.create(new double[]{0, 1}));

        input.putRow(1, Nd4j.create(new double[]{1, 1}));
        outPut.putRow(1, Nd4j.create(new double[]{0, 1}));

        input.putRow(2, Nd4j.create(new double[]{1, 0}));
        outPut.putRow(2, Nd4j.create(new double[]{1, 0}));

        input.putRow(3, Nd4j.create(new double[]{0, 1}));
        outPut.putRow(3, Nd4j.create(new double[]{1, 0}));

        System.out.println(input);
        System.out.println(outPut);
        return new DataSet(input, outPut);
    }
}
