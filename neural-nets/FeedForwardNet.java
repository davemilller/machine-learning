import java.util.*;

/*
    A feed forward neural network implemented with no outside libraries
    Capable of training by
        -backpropagation (default)
        -genetic algorithm
        -particle swarm optimization
        -differential evolution

    @Author: Dave Miller
 */
public class FeedForwardNet {

    //===============================================>LAYER
    /*
        Layer describes a single layer in the network
        It holds the activations sigmoid(W*X+B) for the layer,
        and the weights associated with connections to the
        next layer.
     */
    static class Layer {
        private double[][] weights;
        private double bias;
        private double[] nodes;
        private double[] output;

        public Layer(int size) {
            nodes = new double[size];
        }

        /*
            First pass through, returns feedForward() so you can create the other
            layers. If nextLayerSize == 0, this is the last layer and input is
            returned.
         */
        public double[] init(double[] input, int nextLayerSize) {
            //if last layer return input
            if (nextLayerSize > 0) {
                weights = new double[nextLayerSize][input.length+1];
                output = new double[nextLayerSize];
                bias = 1.0;

                //randomize weights and biases
                for (int row = 0; row < weights.length; row++) {
                    for (int col = 0; col < weights[0].length; col++) {
                        weights[row][col] = (Math.random()* 0.02) -0.01;
                    }
                }
                return feedForward(input);
            }
            else {
                return input;
            }
        }

        /*
            Gets an array of output[] that corresponds to the input
            to the next layer if it exists

            This should not be called on the output layer
         */
        public double[] feedForward(double[] input) {
            this.nodes = input;
            Arrays.fill(this.output, 0);

            //[output] = [weights] * [nodes]
            for (int i= 0; i< weights.length; i++) {
                for (int k= 0; k< weights[0].length-1; k++) {
                    output[i] += (weights[i][k] * input[k]);
                }
            }

            //handle bias
            for (int i= 0; i< weights.length; i++) {
                output[i] += (weights[i][weights[0].length-1] * bias);
            }

            //sigmoid( [output] )
            if (this.output.length > 1) {   //if last layer on regression, use linear activation
                for (int i = 0; i < output.length; i++) {
                    output[i] = sigmoid(output[i]);
                }
            }

            return output;
        }

        /*
            Update weights of this layer given a weightUpdate array of
            length nextLayerSize
         */
        public void updateWeights(double[] weightUpdate, double eta) {
            for (int i= 0; i< weights.length; i++) {
                for (int j= 0; j< weights[0].length-1; j++) {
                    weights[i][j] += (eta * weightUpdate[i] * nodes[j]);
                }
            }

            //bias
            for (int i= 0; i< weights[0].length; i++) {
                weights[weights.length-1][i] += (eta * weightUpdate[weightUpdate.length-1]);
            }
        }

        /*
            Gets weight updates for previous layer according to backprop
         */
        public double[] getWeightUpdates(double[] weightUpdate) {
            double[] nextUpdate = new double[nodes.length+1];
            for (int i= 0; i< weights.length; i++) {
                for (int j= 0; j< weights[0].length-1; j++) {
                    nextUpdate[j] += (weightUpdate[i] * weights[i][j]);
                }
            }

            //bias
            for (int i= 0; i< weights[0].length; i++) {
                nextUpdate[nextUpdate.length-1] += weightUpdate[weightUpdate.length-1] * weights[weights.length-1][i];
            }

            //backprop
            for (int i= 0; i< nextUpdate.length-1; i++) {
                nextUpdate[i] *= nodes[i] * (1 - nodes[i]);
            }

            return nextUpdate;
        }

        /*
            For GA, DE, etc.
         */
        public double[][] getWeights() { return weights; }

        /*
            Assumes weight array is the right size
         */
        public void setWeights(double[][] newWeights) { this.weights = newWeights; }

        /*
            Activation function
         */
        public double sigmoid(double z) {
            return 1 / (1 + Math.exp(-1 * z));
        }
    }


    //========================================>FEED FORWARD NET

    public Layer[] network;
    public DataC[] data;
    public double eta;
    public boolean isClassification;
    public double fitness;

    /*
        Takes an array of layer sizes for the network i.e
        [4, 2, 1] is a 3 layer net with 4 nodes in 1st layer,
        2 in 2nd layer, etc.

        For classification, layers should be [numFeatures, ..., numClasses]
        For regression, layers should be [numFeatures, ..., 1]
     */
    public FeedForwardNet(DataC[] trainingData, int[] layers, boolean isClassification) {
        this.data = trainingData;
        this.eta = 0.5;
        this.isClassification = isClassification;
        this.fitness = 0;
        network = new Layer[layers.length];

        //for each layer, make a Layer object of size layers[i]
        for (int i= 0; i< layers.length; i++) {
            network[i] = new Layer(layers[i]);
        }

        init();
    }
    
    public FeedForwardNet(FeedForwardNet clone) {
    	this.network = clone.network;
    	this.data = clone.data;
    	this.eta = clone.eta;
    	this.isClassification = clone.isClassification;
    	this.fitness = clone.fitness;
    }


    /*
        Calls Layer.init() for each layer in the network
     */
    public void init() {
        //init() first layer with DataC[] input
        double[] output;
        output = network[0].init(data[0].getNormalizedFeatures(), network[1].nodes.length);

        //hidden layer(s)
        for (int i= 1; i< network.length-1; i++) {
            output = network[i].init(output, network[i+1].nodes.length);
        }

        //output layer
        network[network.length-1].init(output, 0);
    }

    /*
        Push 1 data sample through the network,
        returns output layer
     */
    public double[] feedForward(double[] in) {
        //first layer
        double[] output;
        output = network[0].feedForward(in);

        //hidden layer(s)
        for (int i= 1; i< network.length-1; i++) {
            output = network[i].feedForward(output);
        }

        //output layer
        return output;
    }

    /*
        Update weights over #epochs using backpropagation
        Uses mini-batch updating, see updateMiniBatch()
     */
    public void backprop(DataC[] trainingData, int epochs) {

        int miniBatchSize;

        if (isClassification) { miniBatchSize = 4; }
        else { miniBatchSize = data.length / 10; }

        for (int i= 0; i< epochs; i++) {
            double[] weightUpdate = new double[network[network.length-1].nodes.length];
            double totalError = 0;

            List<DataC> temp = Arrays.asList(trainingData);
            Collections.shuffle(temp);

            List<DataC> miniBatch = new ArrayList<>(temp.subList(0, miniBatchSize));

            //get weight update average at output layer for mini batch
            for (DataC d : miniBatch) {
                double[] output = feedForward(d.getNormalizedFeatures());
                double target;
                double[] t;

                //CLASSIFICATION
                if (isClassification) {
                    target = Integer.parseInt(d.getClassLabel()) - 1;
                    t = new double[output.length];
                    Arrays.fill(t, 0);
                    t[(int) target] = 1;
                }
                //REGRESSION
                else {
                    target = Double.parseDouble(d.getClassLabel());
                    t = new double[output.length];
                    t[0] = target;
                }

                totalError += error(output, t);
                double[] deltas = delta(output, t);

                for (int j= 0; j< weightUpdate.length; j++) {
                    weightUpdate[j] += deltas[j];
                }

            }

            totalError /= miniBatchSize;

            //average weight updates
            for (int j= 0; j< weightUpdate.length; j++) {
                weightUpdate[j] /= miniBatchSize;
            }

            updateMiniBatch(weightUpdate);

            System.out.println("Epoch "+ i + " -->Error: " + totalError);
        }
    }

    
    /*
        Updates weights based on performance of mini batch
        Calls Layer.getUpdates() for each layer, then Layer.updateWeights()
     */
    public void updateMiniBatch(double[] weightUpdate) {
        //get updates before updating
        List<double[]> updates = new LinkedList<>();    //need to use for non-rectangular array
        updates.add(weightUpdate);
        double[] prevUpdate = weightUpdate;

        for (int i= network.length-2; i> 0; i--) {
            double[] update = network[i].getWeightUpdates(prevUpdate);
            updates.add(update);
            prevUpdate = update;
        }

        //apply updates
        int c = 0;
        for (int i= network.length-2; i>= 0; i--) {
            network[i].updateWeights(updates.get(c++), eta);
        }
    }

    /*
        Splits data into training (9/10) and testing (1/10).
        Trains w/ training data, then evaluates testing
        data on the network.
     */
    public void evaluate() {
        double totalAvg= 0;

        //setting up training & testing sets
        int trainLen = (int)(0.9 * data.length) + 1;
        int testLen = data.length - trainLen;
        System.out.println("Train: " + trainLen);
        System.out.println("Test:  " + testLen);
        DataC[] train = new DataC[trainLen];
        DataC[] test;
        DataC[][] split = new DataC[10][testLen];

        //shuffle
        List<DataC> temp = Arrays.asList(data);
        Collections.shuffle(temp);
        DataC[] shuffled = new DataC[data.length];
        temp.toArray(shuffled);

        //split into 10 arrays
        int c= 0;
        for (int i= 0; i< 10; i++) {
            for (int j= 0; j< testLen; j++) {
                if (c >= data.length) {
                    split[i][j] = null;
                }
                else {
                    split[i][j] = shuffled[c++];
                }
            }
        }

        //cross-validate 10x
        for (int fold= 0; fold< 10; fold++) {
            int count = 0;
            double error = 0;

            //train fold to (fold+8)%10, test on (fold+9)%10
            List<DataC> list = new ArrayList<>();
            for (int i= 0; i< 9; i++) {
                //add 1/10 of data to list 9 times
                list.addAll(Arrays.asList(split[(i+fold)%10]));
            }

            //fill train[] and test[]
            list.toArray(train);
            test = split[(fold+9)%10];

            for (DataC d : test) {
                double guess;
                double trueClass;
                double[] output = feedForward(d.getNormalizedFeatures());

                //CLASSIFICATION - acts as softmax
                if (isClassification) {
                    trueClass = Integer.parseInt(d.getClassLabel());

                    double max = Double.NEGATIVE_INFINITY;
                    guess = 0;

                    for (int i = 0; i < output.length; i++) {
                        if (output[i] > max) {
                            max = output[i];
                            guess = i + 1;
                        }
                    }
                    if ((int) guess == trueClass) {
                        count++;
                    }
                }
                //REGRESSION
                else {
                    trueClass = Double.parseDouble(d.getClassLabel());
                    guess = output[0];

                    error += Math.pow(guess - trueClass, 2);
                }

//                System.out.println("\tG: " + guess + "\tT: " + trueClass);
            }

            if (isClassification) {
                double accuracy = (double) count / testLen;
                totalAvg += accuracy;
                System.out.println(count + " / " + testLen + " = " + accuracy);
            } else {
                double MSE = error / testLen;
                totalAvg += MSE;
                System.out.println("MSE: " + MSE);
            }
        }
        totalAvg /= 10;

        if (isClassification) { System.out.println("Total accuracy: " + totalAvg); }
        else { System.out.println("Total MSE: " + totalAvg); }
    }

    /*
        Gives squared error
     */
    public double error(double[] output, double[] target) {
        double sum= 0;

        for (int i= 0; i< output.length; i++) {
            sum += Math.pow(target[i] - output[i], 2);
        }

        return sum / 2;
    }

    /*
        Gives squared error derivative DELTA_j
     */
    public double[] delta(double[] output, double[] target) {
        double[] delta = new double[output.length];

        for (int i= 0; i< output.length; i++) {
            delta[i] = (target[i] - output[i]);
        }

        return delta;
    }

    public List<double[][]> getWeights() {
        List<double[][]> weights = new ArrayList<>();
        for (int i= 0; i< network.length-1; i++) {
            weights.add(network[i].getWeights());
        }

        return weights;
    }

    public void setWeights(List<double[][]> newWeights) {
        for(int i= 0; i< network.length-1; i++) {
            network[i].setWeights(newWeights.get(i));
        }
    }

    public void updateFitness() {
        double totalError = 0;

        for (DataC d : data) {
            double[] output = feedForward(d.getNormalizedFeatures());
            double target;
            double[] t;

            //CLASSIFICATION
            if (isClassification) {
                target = Integer.parseInt(d.getClassLabel()) - 1;
                t = new double[output.length];
                Arrays.fill(t, 0);
                t[(int) target] = 1;
            }
            //REGRESSION
            else {
                target = Double.parseDouble(d.getClassLabel());
                t = new double[output.length];
                t[0] = target;
            }

            totalError += error(output, t);

        }

        totalError /= data.length;

        this.fitness = totalError;
    }
    


    //==================================>MAIN
    public static void main(String[] args) {

        //---------------------------------->GLASS
        DataSetUp glass = new DataSetUp("data-sets/glass.data", "end", "classification");
        glass.zScoreNormalize();

        int inputLen = glass.getAllData()[0].getNormalizedFeatures().length;
        int[] layers = {inputLen, 12, 12, 7};
//        proj4.FeedForwardNet network = new proj4.FeedForwardNet(glass.getAllData(), layers, true);
//        network.evaluate();

        //---------------------------------->SOYBEAN
        DataSetUp soybean = new DataSetUp("data-sets/soybean-small.data", "endS", "classification");
        soybean.zScoreNormalize();

        inputLen = soybean.getAllData()[0].getNormalizedFeatures().length;
        int[] layers3 = {inputLen, 12, 4};
        FeedForwardNet netS = new FeedForwardNet(soybean.getAllData(), layers3, true);
        netS.backprop(soybean.data, 10000);
        netS.evaluate();

        //---------------------------------->BREAST CANCER
        DataSetUp breastCancer = new DataSetUp("data-sets/breast-cancer-wisconsin.data", "endB", "classification");
        breastCancer.zScoreNormalize();

        inputLen = breastCancer.getAllData()[0].getNormalizedFeatures().length;
        int[] layersB = {inputLen, 10, 10, 2};
//        proj4.FeedForwardNet netB = new proj4.FeedForwardNet(breastCancer.getAllData(), layersB, true);
//        netB.evaluate();

        //---------------------------------->FOREST FIRES
        DataSetUp forestFires = new DataSetUp("data-sets/forestfires.data", "endF", "regression");
        forestFires.zScoreNormalize();

        inputLen = forestFires.getAllData()[0].getNormalizedFeatures().length;
        int[] layersF = {inputLen, 16, 1};
//        proj4.FeedForwardNet net = new proj4.FeedForwardNet(forestFires.getAllData(), layersF, false);
//        net.evaluate();

        //---------------------------------->ABALONE
        DataSetUp abalone = new DataSetUp("data-sets/abalone.data", "endA", "regression");
        abalone.zScoreNormalize();

        inputLen = abalone.getAllData()[0].getNormalizedFeatures().length;
        int[] layersA = {inputLen, 20, 1};
//        proj4.FeedForwardNet netA = new proj4.FeedForwardNet(abalone.getAllData(), layersA, false);
//        netA.evaluate();

        //---------------------------------->MACHINE
        DataSetUp machine = new DataSetUp("data-sets/machine.data", "end", "regression");
        machine.zScoreNormalize();

        inputLen = machine.getAllData()[0].getNormalizedFeatures().length;
        int[] layersM = {inputLen, 16, 16, 1};
//        proj4.FeedForwardNet netM = new proj4.FeedForwardNet(machine.getAllData(), layersM, false);
//        netM.evaluate();
    }

}
