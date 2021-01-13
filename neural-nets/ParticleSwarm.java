

import java.util.*;

/*
    Each particle has:
        position -> weights
        velocity -> weight update
        pBest
        w, c1, c2

    Each swarm has:
        particles[]
        lBest[]
 */
public class ParticleSwarm {

    //========================================================>Particle

    static class Particle {

        public FeedForwardNet position;
        public List<double[][]> velocity;
        public FeedForwardNet pBest;
        public double pBestError;
        public double w, c1, c2;
        public int local;

        public Particle(DataC[] trainingData, int[] layers, boolean isClassification, double w, double c1, double c2, int local) {
            position = new FeedForwardNet(trainingData, layers, isClassification);
            pBestError = Double.POSITIVE_INFINITY;
            pBest = new FeedForwardNet(trainingData, layers, isClassification);
            velocity = position.getWeights();
            this.w = w;
            this.c1 = c1;
            this.c2 = c2;
            this.local = local;
        }


        /*
            v(t+1) = wv(t) + c1r1(pBest - position) + c2r2(gBest - position)
         */
        public void updateVelocity(List<double[][]> gBest) {
            double r1, r2;
            List<double[][]> pBestWeights = pBest.getWeights();
            List<double[][]> pWeights = position.getWeights();
            List<double[][]> newVelocity = new ArrayList<>();

            int c= 0;
            for(double[][] layer : velocity) {
                double[][] v1 = new double[layer.length][layer[0].length];

                for(int i= 0; i< layer.length; i++) {
                    for(int j= 0; j< layer[0].length; j++) {
                        r1 = Math.random();
                        r2 = Math.random();

                        v1[i][j] = w * velocity.get(c)[i][j] +
                                    c1 * r1 * (pBestWeights.get(c)[i][j] - pWeights.get(c)[i][j]) +
                                    c2 * r2 * (gBest.get(c)[i][j] - pWeights.get(c)[i][j]);
                    }
                }

                newVelocity.add(v1);
                c++;
            }

            velocity = newVelocity;
        }


        /*
            x(t+1) = x(t) + v(t)
         */
        public void updatePosition() {
            List<double[][]> newWeights = new ArrayList<>();

            int c= 0;
            for(double[][] layer: position.getWeights()) {
                double[][] w1 = new double[layer.length][layer[0].length];

                for(int i= 0; i< layer.length; i++) {
                    for(int j= 0; j< layer[0].length; j++) {
                        w1[i][j] = layer[i][j] + velocity.get(c)[i][j];
                    }
                }

                newWeights.add(w1);
                c++;
            }

            position.setWeights(newWeights);
            position.updateFitness();

            if (position.fitness < pBestError) {
                pBestError = position.fitness;
                pBest.setWeights(position.getWeights());
            }
        }

    }


    //========================================================>ParticleSwarm

    public Particle[] particles;
    public FeedForwardNet[] lBest;
    public double gBestError;

    public ParticleSwarm(int populationSize, DataC[] trainingData, int[] layers, boolean isClassification) {
        particles = new Particle[populationSize];
        gBestError = Double.POSITIVE_INFINITY;
        double w = 0.1;
        double c1 = 3;
        double c2 = 1.5;

        lBest = new FeedForwardNet[populationSize / 2];
        for(int i= 0; i< lBest.length; i++) {
            lBest[i] = new FeedForwardNet(trainingData, layers, isClassification);
        }

        int local = (int)(Math.random() * (lBest.length-1));
        for(int i= 0; i< populationSize; i++) {
            particles[i] = new Particle(trainingData, layers, isClassification, w, c1, c2, local);
        }
    }


    public void updateVelocities() {
        for (Particle particle : particles) {
            particle.updateVelocity(lBest[particle.local].getWeights());
        }
    }


    /*
        Update lBest if necessary
     */
    public void updatePositions() {
        for (Particle particle : particles) {
            particle.updatePosition();

            if (particle.pBestError < lBest[particle.local].fitness) {
                gBestError = particle.pBestError;
                lBest[particle.local].setWeights(particle.pBest.getWeights());
                lBest[particle.local].updateFitness();
            }
        }

        //give each lBest a chance to check neighbor
        for (int i= 0; i< lBest.length; i++) {
            if (lBest[(i+1)% lBest.length].fitness < lBest[i].fitness) {
                lBest[i].setWeights(lBest[(i+1)% lBest.length].getWeights());
            }
        }
    }


    /*
        print gBest
     */
    public void evaluate() {
        System.out.print("[");
        for(Particle p : particles) {
            System.out.printf("%.3f ", p.pBestError);
        }

        for (FeedForwardNet net : lBest) {
            System.out.printf("[%.3f]", net.fitness);
        }
        System.out.print("]\n");

        //sort by fitness
        Arrays.sort(particles, new Comparator<Particle>() {
            @Override
            public int compare(Particle o1, Particle o2) {
                Double fitness1 = o1.pBestError;
                Double fitness2 = o2.pBestError;
                return fitness1.compareTo(fitness2);
            }
        });
    }


    /*
        1. Update velocities
        2. Update positions
        3. Check status
        4. Repeat for n generations OR until stopping criterion
     */
    public void PSO(int generations) {

        for(int i= 0; i< generations; i++) {

            updateVelocities();

            updatePositions();

            evaluate();

            if (particles[0].pBestError < 0.01) { System.out.println("Generations: "+i); break; }

        }

    }

}
