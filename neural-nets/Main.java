import java.util.*;

/*
    A main driver for all training methods and data sets

    @Author: Dave Miller
 */
public class Main {

    public static void main(String[] args) {

        //--------Loading in data
        ArrayList<DataSetUp> dataSets = new ArrayList<>();
        int[] outputLengths = {7, 4, 2, 1, 1, 1};
        boolean[] isCl = {true, true, true, false, false, false};

        DataSetUp glass = new DataSetUp("C:/Users/super/OneDrive/Desktop/machine-learning/neural-nets/data-sets/glass.data", "end", "classification");
        glass.zScoreNormalize();
        dataSets.add(glass);

        DataSetUp soybean = new DataSetUp("C:/Users/super/OneDrive/Desktop/machine-learning/neural-nets/data-sets/soybean-small.data", "endS", "classification");
        soybean.zScoreNormalize();
        dataSets.add(soybean);

        DataSetUp breastCancer = new DataSetUp("C:/Users/super/OneDrive/Desktop/machine-learning/neural-nets/data-sets/breast-cancer-wisconsin.data", "endB", "classification");
        breastCancer.zScoreNormalize();
        dataSets.add(breastCancer);

        DataSetUp forestFires = new DataSetUp("C:/Users/super/OneDrive/Desktop/machine-learning/neural-nets/data-sets/forestfires.data", "endF", "regression");
        forestFires.zScoreNormalize();
        dataSets.add(forestFires);

        DataSetUp abalone = new DataSetUp("C:/Users/super/OneDrive/Desktop/machine-learning/neural-nets/data-sets/abalone.data", "endA", "regression");
        abalone.zScoreNormalize();
        dataSets.add(abalone);

        DataSetUp machine = new DataSetUp("C:/Users/super/OneDrive/Desktop/machine-learning/neural-nets/data-sets/machine.data", "end", "regression");
        machine.zScoreNormalize();
        dataSets.add(machine);


        //-----text interface
        Scanner in = new Scanner(System.in);
        System.out.println("<----Welcome to neural nets 1.0---->");
        System.out.println("Choose a data set:\n\t(1)Glass\n\t(2)Soybean\n\t(3)Breast cancer\n\t(4)Forest fires\n\t(5)Abalone\n\t(6)Machine\n");
        int dataSet = in.nextInt();

        System.out.println("Choose a training method:\n\t(1)Backpropagation\n\t(2)Genetic algorithm\n\t(3)Particle swarm optimization\n\t(4)Differential evolution\n");
        int trainingMethod = in.nextInt();

        int[] layers = {dataSets.get(dataSet).data[0].getFeatures().length, 16, outputLengths[dataSet]};

        System.out.println("Training...");
        switch (trainingMethod) {
            case 1 -> {
                FeedForwardNet net = new FeedForwardNet(dataSets.get(dataSet).data, layers, isCl[dataSet]);
                net.backprop(dataSets.get(dataSet).data, 10000);
                net.evaluate();
            }
            case 2 -> {
                Genetic G = new Genetic(20, dataSets.get(dataSet).data, layers, isCl[dataSet]);
                G.GA(2000);
                G.population[0].evaluate();
            }
            case 3 -> {
                ParticleSwarm P = new ParticleSwarm(20, dataSets.get(dataSet).data, layers, isCl[dataSet]);
                P.PSO(5000);
                P.particles[0].pBest.evaluate();
            }
            case 4 -> {
                DE D = new DE(20, dataSets.get(dataSet).data, layers, isCl[dataSet]);
                D.DiffEvolution(2000);
                D.population[0].evaluate();
            }
            default -> System.out.println("Bad input");
        }

        System.out.println("Thanks for playing!");
    }

}
