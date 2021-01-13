import java.util.*;

public class DE {
	// skip selection, considering all examples
	// mutation -> crossover -> replacement
	// Mutation is pretty different from GA!
	// Beta used in mutation, p_r is crossover probability for binomial crossover
	// Create and initialize population just like GA
	// While loop
	//    for each individual
	//		  the individual is the target vector
	//        target used to create trial vector, by applying mutation
	//        create an offspring, by applying crossover (only 1 offspring)
	//        if offspring is better than the target
	//            add to pop
    //		  else
	//            add target to pop
	
	
    public FeedForwardNet[] population;
    public double mutationRate;
    public double beta;
    public double crossoverProb;

	/*
	 * Init pop like GA, and start while loop and for loop to go through each individual in pop and do mut->cross->rep until done
	 */
	public DE(int populationSize, DataC[] trainingData, int[] layers, boolean isClassification) {
        population = new FeedForwardNet[populationSize];
        
        //all of this is hard coded rn
        mutationRate = 0.05;
        beta = .1;
        crossoverProb = .5; //makes it the same as GA

        for (int i= 0; i< populationSize; i++) {
            population[i] = new FeedForwardNet(trainingData, layers, isClassification);
        }
	}
	
	
	public FeedForwardNet mutation(FeedForwardNet targetVector) {
		//Done?
		//target vector x_j
		//select three other distinct vectors from out population X1,X2,X3
		//calculate trail vector u_j(u_i?) = X1 + Beta(X2-X3)            Beta in [0,2]
		      //X2 - X3 reverses the direction of the X2 vector? (should it be X3-X2?)
		//return the trial vector

		Set<FeedForwardNet> distinct = new HashSet<>();

		while(distinct.size() < 3) {
			distinct.add(population[(int)(Math.random() * (population.length-1))]);
		}

		FeedForwardNet[] distinctVectors = new FeedForwardNet[3];
		distinct.toArray(distinctVectors);

		List<double[][]> X1 = distinctVectors[0].getWeights();
		List<double[][]> X2 = distinctVectors[1].getWeights();
		List<double[][]> X3 = distinctVectors[2].getWeights();

		List<double[][]> trialWeights = new ArrayList<>();
		
		int z = 0;
		for(double[][] layer : X1) {
			double[][] newWeights = new double[layer.length][layer[0].length];
			for(int i= 0; i< layer.length; i++) {
                for(int j= 0; j< layer[0].length; j++) {
                	newWeights[i][j] = X1.get(z)[i][j] + (beta*(X2.get(z)[i][j] - X3.get(z)[i][j]));
                }
			}

			trialWeights.add(newWeights);
			z++;
		}
		
		//Changing weights
		targetVector.setWeights(trialWeights);
		return targetVector;
	}
	
	public FeedForwardNet crossover(FeedForwardNet targetVector, FeedForwardNet trialVector) {
		//Done?
		//Binomial Crossover
		//x_j'_i (where i are the individual components of the vector)
		//
		//		   {x_j_i if rand(0,1) <= p_r (if p_r is .5 then its the same as GA
		//x_j'_i = {
		//		   {U_j_i otherwise
		//
		//basically for loop over the components of the vector and pick as described above if the 
		//   component should be taken from x_j or from u_j
		// then return offspring x_j'
		
		
		List<double[][]> targetWeights = targetVector.getWeights();
		List<double[][]> trialWeights = trialVector.getWeights();
		
		List<double[][]> offspringWeights = new ArrayList<>();
		
		int z = 0;
		for(double[][] layer : targetWeights) {
			double[][] newWeights = new double[layer.length][layer[0].length];
			for(int i= 0; i< layer.length; i++) {
                for(int j= 0; j< layer[0].length; j++) {
        			if(Math.random() <= crossoverProb) {
        				newWeights[i][j] = targetWeights.get(z)[i][j];
        			}
        			else {
        				newWeights[i][j] = trialWeights.get(z)[i][j];
        			}
                }
			}
			offspringWeights.add(newWeights);
			z++;
		}
		
		targetVector.setWeights(offspringWeights);
		return targetVector;
	}
	
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

	public void DiffEvolution(int generations) {
    	// While loop (idk what condition needs to be ^-^)
    	//    for each individual
    	//		  the individual is the target vector
    	//        target used to create trial vector, by applying mutation
    	//        create an offspring, by applying crossover (only 1 offspring)
    	//        if offspring is better than the target
    	//            add to pop
        //		  else
    	//            add target to pop
		
		//go until we go the number of generations or until specific condition is met
        for(int z = 0; z < generations; z++) {
        	
        	//for each individual FFN
        	for(int i = 0; i < population.length; i++) {
        		
        		FeedForwardNet targetVector = new FeedForwardNet(population[i]);

        		//Evaluate Fitness
        		double targetFitness = targetVector.fitness;
        		
        		//Mutation
        		FeedForwardNet tempTrial = mutation(targetVector);
        		
        		//Offspring (crossover)
				FeedForwardNet trialVector = crossover(targetVector,tempTrial);
        		trialVector.updateFitness();
        		
        		double offspringFitness = trialVector.fitness;
        		
        		if(offspringFitness < targetFitness) {
        			population[i].setWeights(trialVector.getWeights());
        		}

        	}
        	evaluate();

        	if (population[0].fitness < 0.008) { System.out.println("Generations: "+ z); break; }
        }
        
	}

}
