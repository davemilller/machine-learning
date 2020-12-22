

import java.util.*;

public class Genetic {

    public FeedForwardNet[] population;
    public double mutationRate;


    /*
        Init population
     */
    public Genetic(int populationSize, DataC[] trainingData, int[] layers, boolean isClassification) {
        population = new FeedForwardNet[populationSize];
        mutationRate = 0.05;

        for (int i= 0; i< populationSize; i++) {
            population[i] = new FeedForwardNet(trainingData, layers, isClassification);
        }

        evaluate();
    }


    /*
        Tournament selection, k=2 for now
        Choosing population.length / 2 individuals
     */
    public FeedForwardNet[] selection() {
        int k = 2;
        int rand;
        Set<FeedForwardNet> finalSelection = new HashSet<>();   //to select w/o replacement

        for (int i= 0; i< population.length / 2; i++) {
            FeedForwardNet[] tourney = new FeedForwardNet[k];

            //select k individuals at random
            for (int j= 0; j< k; j++) {
                rand = (int)(Math.random() * (population.length -1));
                tourney[j] = population[rand];
            }

            //add best individual to finalSelection[]
            double best= Double.POSITIVE_INFINITY;
            FeedForwardNet bestInd = null;
            for (int j= 0; j< k; j++) {
                if (tourney[j].fitness < best) {
                    bestInd = tourney[j];
                    best = tourney[j].fitness;
                }
            }

            finalSelection.add(bestInd);
        }

        //make sure of even number of parents
        int i= 0;
        while (finalSelection.size() % 2 == 1) {
            finalSelection.add(population[i++]);
        }

        FeedForwardNet[] selection = new FeedForwardNet[finalSelection.size()];
        finalSelection.toArray(selection);

        return selection;
    }


    /*
        Uniform crossover
        2 parents -> 2 children

        Decided to do mutation in here since already accessing
        the double arrays,
        currently hardcoded mutation rate of 0.1,
        adds to value a number in range [-0.1, +0.1]
     */
    public FeedForwardNet[] crossover(FeedForwardNet[] selected) {
        double rand;

        //goes in groups of 2
        for(int i= 0; i< selected.length; i++) {
            List<double[][]> parent1 = selected[i].getWeights();
            List<double[][]> parent2 = selected[i+1].getWeights();

            List<double[][]> child1 = new ArrayList<>();
            List<double[][]> child2 = new ArrayList<>();

            int c = 0;
            for(double[][] layer: parent1) {
                double[][] c1 = new double[layer.length][layer[0].length];
                double[][] c2 = new double[layer.length][layer[0].length];

                for(int j= 0; j< layer.length; j++) {
                    for(int k= 0; k< layer[0].length; k++) {

                        //CROSSOVER
                        rand = Math.random();

                        if (rand >= 0.5) {
                            c2[j][k] = parent1.get(c)[j][k];
                            c1[j][k] = parent2.get(c)[j][k];
                        }
                        else {
                            c1[j][k] = parent1.get(c)[j][k];
                            c2[j][k] = parent2.get(c)[j][k];
                        }

                        //MUTATION
                        rand = Math.random();

                        if (rand >= (1 - mutationRate)) {
                            rand = (Math.random() * .2)-0.1;
                            c1[j][k] += rand;
                        }

                        rand = Math.random();

                        if (rand >= (1 - mutationRate)) {
                            rand = (Math.random() * .2)-0.1;
                            c2[j][k] += rand;
                        }
                    }
                }

                child1.add(c1);
                child2.add(c2);
                c++;
            }

            selected[i].setWeights(child1);
            selected[i+1].setWeights(child2);

            i++;
        }

        return selected;
    }


    /*
        Replacement via steady state - kill weakest & replace

        Assumes fitness sorted population best fitness to worst
     */
    public void replace(FeedForwardNet[] newIndividuals) {
        int j= population.length-1;
        for (FeedForwardNet newIndividual : newIndividuals) {
            population[j--].setWeights(newIndividual.getWeights());
        }
    }


    /*
        Check progress of GA, update fitnesses,
        maybe sort the population by fitness?
     */
    public void evaluate() {
        //update fitness
        for (FeedForwardNet feedForwardNet : population) {
            feedForwardNet.updateFitness();
        }

        //sort by fitness
        Arrays.sort(population, new Comparator<FeedForwardNet>() {
            @Override
            public int compare(FeedForwardNet o1, FeedForwardNet o2) {
                Double fitness1 = o1.fitness;
                Double fitness2 = o2.fitness;
                return fitness1.compareTo(fitness2);
            }
        });

        System.out.print("[");
        for (FeedForwardNet net: population) {
            System.out.printf("%.3f, ", net.fitness);
        }
        System.out.print("]\n");
    }


    /*
        1. Make a randomized population & gather fitness scores
        2. Selection
        3. Crossover & mutation
        4. Replacement
        5. Repeat for n generations OR until stopping criterion
     */
    public void GA(int generations) {

        for (int i= 0; i< generations; i++) {
            FeedForwardNet[] selected = selection();

            FeedForwardNet[] children = crossover(selected);

            replace(children);

            evaluate();

            if (population[0].fitness < 0.01) { System.out.println("Generations: "+i); break; }
        }

    }


    public static void main(String[] args) {
        //---------------------------------->GLASS
        DataSetUp glass = new DataSetUp("data-sets/glass.data", "end", "classification");
        glass.zScoreNormalize();

        int inputLen = glass.getAllData()[0].getNormalizedFeatures().length;
        int[] layers = {inputLen, 7};

//        Genetic G1 = new Genetic(30, glass.getAllData(), layers, true);
//        G1.GA(20000);
//
//        G1.population[0].evaluate();

        //---------------------------------->SOYBEAN
        DataSetUp soybean = new DataSetUp("data-sets/soybean-small.data", "endS", "classification");
        soybean.zScoreNormalize();

        inputLen = soybean.getAllData()[0].getNormalizedFeatures().length;
        int[] layersS = {inputLen, 12, 12, 4};

//        Genetic G2 = new Genetic(20, soybean.getAllData(), layersS, true);
//        G2.GA(2000);
//
//        G2.population[0].evaluate();

        //---------------------------------->BREAST CANCER
        DataSetUp breastCancer = new DataSetUp("data-sets/breast-cancer-wisconsin.data", "endB", "classification");
        breastCancer.zScoreNormalize();

        inputLen = breastCancer.getAllData()[0].getNormalizedFeatures().length;
        int[] layersB = {inputLen, 2};

//        Genetic G3 = new Genetic(20, breastCancer.getAllData(), layersB, true);
//        G3.GA(5000);
//
//        G3.population[0].evaluate();

        //---------------------------------->FOREST FIRES
        DataSetUp forestFires = new DataSetUp("data-sets/forestfires.data", "endF", "regression");
        forestFires.zScoreNormalize();

        inputLen = forestFires.getAllData()[0].getNormalizedFeatures().length;
        int[] layersF = {inputLen, 12, 1};

//        Genetic G4 = new Genetic(20, forestFires.getAllData(), layersF, false);
//        G4.GA(2000);
//
//        G4.population[0].evaluate();

        //---------------------------------->ABALONE
        DataSetUp abalone = new DataSetUp("data-sets/abalone.data", "endA", "regression");
        abalone.zScoreNormalize();

        inputLen = abalone.getAllData()[0].getNormalizedFeatures().length;
        int[] layersA = {inputLen, 1};

//        Genetic G5 = new Genetic(20, abalone.getAllData(), layersA, false);
//        G5.GA(2000);
//
//        G5.population[0].evaluate();

        //---------------------------------->MACHINE
        DataSetUp machine = new DataSetUp("data-sets/machine.data", "end", "regression");
        machine.zScoreNormalize();

        inputLen = machine.getAllData()[0].getNormalizedFeatures().length;
        int[] layersM = {inputLen, 1};

//        Genetic G6 = new Genetic(20, machine.getAllData(), layersM, false);
//        G6.GA(2000);
//
//        G6.population[0].evaluate();
    }

}
