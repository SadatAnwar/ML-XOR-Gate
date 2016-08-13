package neuralnetwork;

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

import java.util.Random;

public class SquareRoot
{
    private static final int OUT_NEURONS = 1;

    private static final int HIDDEN_NEURONS = 5;

    private static final int IN_NEURONS = 1;

    private static final int MAX = 10000;

    private static final int NUMBER_EPOCHS = 1;

    public static void main(String... args)
    {
        new SquareRoot();
    }

    public SquareRoot()
    {
        DataSetIterator iterator = new TestDataSetIterator(generateTrainingData(100000, new Random(12345L)), 500);

        MultiLayerNetwork neuralNetwork = getNeuralNetwork();
        neuralNetwork.setListeners(new ScoreIterationListener(100));
        neuralNetwork.init();

        trainNetwork(neuralNetwork, iterator);

        // Test network performance
        Random rand = new Random(System.currentTimeMillis());
        for (int i = 0; i < 10; i++) {
            int randomNumber = rand.nextInt((MAX- 1) + 1) + 5000*i;
            double squareRoot = Math.sqrt(randomNumber);
            INDArray normalizedOutput = neuralNetwork.output(normalizeInput(randomNumber));
            double output = denormalizeOutput(normalizedOutput);

            System.out.println(String.format("Input: [%s] ExpectedSqrt: [%s] ActualSqrt [%s] Error [%s]", randomNumber, squareRoot, output, (Math.abs(squareRoot-output))));
        }
    }

    private double denormalizeOutput(INDArray normalizedOutput)
    {
        double data = normalizedOutput.getDouble(0,0);
        return data*Math.sqrt(MAX);
    }

    private void trainNetwork(MultiLayerNetwork neuralNetwork, DataSetIterator iterator)
    {
        for (int i = 0; i < NUMBER_EPOCHS; i++) {
            iterator.reset();
            neuralNetwork.fit(iterator);
        }

    }

    private MultiLayerNetwork getNeuralNetwork()
    {
        return new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(123)
            .iterations(1000)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(0.1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(IN_NEURONS).nOut(HIDDEN_NEURONS)
                .activation("tanh")
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation("identity")
                .nIn(HIDDEN_NEURONS).nOut(OUT_NEURONS).build())
            .pretrain(false)
            .backprop(true)
            .build()
        );
    }

    /**
     * A helper method that will create a training dataSet
     *
     * Since the Neural Network requires the data to be normalized withing the range of the activation function,
     * its important we normalize the input and the outputs. The normalization of Inouts and Outputs can be done using
     * different scale, the important thing here is that the scale remain the same during de-normalization
     *
     * @param dataSetSize define the number of data points to be present in the data set
     *
     * @return a {@link DataSet} object that can be used to get a {@link org.nd4j.linalg.dataset.api.iterator.DataSetIterator}
     */
    private DataSet generateTrainingData(int dataSetSize, Random rand)
    {
        INDArray input = Nd4j.zeros(dataSetSize, 1);
        INDArray output = Nd4j.zeros(dataSetSize, 1);
        for (int i = 0; i < dataSetSize; i++) {
            int randomNumber = rand.nextInt((MAX - 1) + 1);
            double squareRoot = Math.sqrt(randomNumber);
            input.putRow(i, normalizeInput(randomNumber));
            output.putRow(i, normalizeTarget(squareRoot));

        }
        return new DataSet(input, output);
    }

    private INDArray normalizeTarget(double squareRoot)
    {
        return Nd4j.create(new double[]{squareRoot/Math.sqrt(MAX)});
    }

    private INDArray normalizeInput(double randomNumber)
    {
        return Nd4j.create(new double[]{randomNumber/MAX});
    }
}
