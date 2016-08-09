import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class XorGate
{
    public static final int COLS = 2;

    public static final int ROWS = 4;

    public static void main(String... args)
    {
        new XorGate();
    }

    public XorGate()
    {
        MultiLayerNetwork nn = getNeuralNetwork();
        nn.init();
        nn.setListeners(new ScoreIterationListener(100));
        DataSet trainingData = getTrainingData();
        DataSetIterator iterator = new SamplingDataSetIterator(trainingData, trainingData.numExamples(), trainingData.numExamples());

        for (int i = 0; i < 10; i++) {
            iterator.reset();
            nn.fit(iterator);
        }

        //Test the network here

        INDArray testInputs = Nd4j.create(new double[] { 1, 1 });
        System.out.println(nn.output(testInputs).toString());

        testInputs = Nd4j.create(new double[] { 0, 0 });
        System.out.println(nn.output(testInputs).toString());

        testInputs = Nd4j.create(new double[] { 0, 1 });
        System.out.println(nn.output(testInputs).toString());

        testInputs = Nd4j.create(new double[] { 1, 0 });
        System.out.println(nn.output(testInputs).toString());
    }

    private MultiLayerNetwork getNeuralNetwork()
    {
        int numInput = 2;
        int numOutputs = 2;
        int nHidden = 2;
        return new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(123)
            .iterations(1000)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(0.1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                .activation("sigmoid")
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                .activation("sigmoid")
                .nIn(nHidden).nOut(numOutputs).build())
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
    private DataSet getTrainingData()
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
