package project_2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/*
    Class w/ static methods to give
        k-means clusters
        k-medoids clusters

    Both methods return a reduced set of new DataC objects
    with the feature vectors corresponding to the means
    and medoids, respectively.
 */
public class Cluster {

    public Cluster() { }

    public static DataC[] kMeansClusters(DataC[] dataSet, int k) {
        //k is the number of clusters not the k from KNN
        //double[] means =  new double[k];
        double[][] means = new double[k][dataSet[0].getFeatures().length];
        double[][] newMeans =  new double[k][dataSet[0].getFeatures().length];

        double min = Double.POSITIVE_INFINITY;
        double max = 0;
        for(DataC data : dataSet) {
            double[] features = data.getFeatures();
            for(double d : features) {
                if(d < min) {
                    min = d;
                }else if(d > max) {
                    max = d;
                }
            }
        }

        for(int i = 0; i < means.length; i++) {
            //randomly initialize means from the data space
            //Unformed sampling
            for(int j = 0; j < dataSet[0].numFeatures(); j++) {
                Random r = new Random();
                double ranNum = r.nextInt((int)((max-min)+1)) + min;
                means[i][j] = ranNum;
            }
        }

        boolean first = true;
        int iterations = 0;

        //REPEAT until no change in means
        while (isDifferent(means, newMeans) || iterations > 100) {
            if(!first) {
                //set means = newMeans after the first time through
                for(int i = 0; i < means.length; i++) {
                    if (dataSet[0].numFeatures() >= 0)
                        System.arraycopy(newMeans[i], 0, means[i], 0, dataSet[0].numFeatures());
                }
            }

            for (DataC d: dataSet) {
                double[] distArray = new double[k];
                double minDist = Double.POSITIVE_INFINITY;
                int minCluster = 1;

                //get distances between each mean and features of d
                for (int i= 0; i< k; i++) {
                    distArray[i] = getDistance(means[i], d.getFeatures());
                    if (distArray[i] < minDist) {
                        minDist = distArray[i];
                        minCluster = i;
                    }
                }

                d.setClusterID(minCluster);

            }

            //calculate new means --> update step
            for (int i= 0; i< k; i++) {
                DataC[] cluster = getCluster(dataSet, i);
                int len = cluster.length;

                //add all features for that cluster to newMeans
                for (DataC d: cluster) {
                    double[] features = d.getFeatures();
                    for (int j= 0; j< features.length; j++) {
                        newMeans[i][j] += features[j];
                    }
                }

                //average over all points in the cluster
                for (int j= 0; j< dataSet[0].getFeatures().length; j++) {
                    if (newMeans[i][j] != 0) {
                        newMeans[i][j] /= len;
                    }
                }
            }

//            for (int j = 0; j< k; j++) {
//                System.out.println(Arrays.toString(newMeans[j]));
//            }
//            System.out.println("-------------");

            first = false;
            iterations++;
        }

        //return new DataC objects
        DataC[] centroids = new DataC[k];
        for (int i= 0; i< k; i++) {
            centroids[i] = new DataC(means[i], i);
        }

//        System.out.println(iterations);

        return centroids;
    }

    public static DataC[] kMedoidsClusters(DataC[] dataSet, int k) {
        double[][] medoids = new double[k][dataSet[0].getFeatures().length];
        double[][] newMedoids =  new double[k][dataSet[0].getFeatures().length];

        double min = Double.POSITIVE_INFINITY;
        double max = 0;
        for(DataC data : dataSet) {
            double[] features = data.getFeatures();
            for(double d : features) {
                if(d < min) {
                    min = d;
                }else if(d > max) {
                    max = d;
                }
            }
        }

        for(int i = 0; i < medoids.length; i++) {
            //randomly initialize medoids from the data space
            for(int j = 0; j < dataSet[0].numFeatures(); j++) {
                Random r = new Random();
                double ranNum = r.nextInt((int)((max-min)+1)) + min;
                medoids[i][j] = ranNum;
            }
        }

        boolean first = true;
        int iterations = 0;

        //REPEAT until no change in medoids
        while (isDifferent(medoids, newMedoids) || iterations > 100) {
            if(!first) {
                //set medoids = newMedoids after the first time through
                for(int i = 0; i < medoids.length; i++) {
                    if (dataSet[0].numFeatures() >= 0)
                        System.arraycopy(newMedoids[i], 0, medoids[i], 0, dataSet[0].numFeatures());
                }
            }

            for (DataC d: dataSet) {
                double[] distArray = new double[k];
                double minDist = Double.POSITIVE_INFINITY;
                int minCluster = 1;

                //get distances between each mean and features of d
                for (int i= 0; i< k; i++) {
                    distArray[i] = getDistance(medoids[i], d.getFeatures());
                    if (distArray[i] < minDist) {
                        minDist = distArray[i];
                        minCluster = i;
                    }
                }

                d.setClusterID(minCluster);

            }

            //update step
            //if there are m points in a cluster, swap the prev centroid with all
            //other m-1 points and make the new centroid the one with min loss --> min
            //sum of dists to all other points in the cluster
            for (int i= 0; i< k; i++) {
                DataC[] cluster = getCluster(dataSet, i);
                double minDistSum = Double.POSITIVE_INFINITY;
                double[] minCentroid = medoids[i];

                for (DataC d: cluster) {
                    double[] newCentroid = d.getFeatures();
                    double distSum= 0;

                    for (DataC d1: cluster) {
                        distSum += getDistance(newCentroid, d1.getFeatures());
                    }

                    if (distSum < minDistSum) {
                        minDistSum = distSum;
                        minCentroid = newCentroid;
                    }
                }

                newMedoids[i] = minCentroid;
            }

//            for (int j = 0; j< k; j++) {
//                System.out.println(Arrays.toString(newMedoids[j]));
//            }
//            System.out.println("-------------");

            first = false;
            iterations++;
        }

        //return new DataC objects
        DataC[] centroids = new DataC[k];
        for (int i= 0; i< k; i++) {
            centroids[i] = new DataC(medoids[i], i);
        }

//        System.out.println(iterations);

        return centroids;
    }

    public static double getDistance(double[] x, double[] y) {
        //error checking - x and y need to have same dimensionality
        if (x.length != y.length) {
            return -1;
        }

        double sum = 0;
        for (int i = 0; i< x.length; i++) {
            sum += Math.pow((x[i] - y[i]), 2);
        }

        return Math.pow(sum, 1.0/2);
    }

    public static boolean isDifferent(double[][] means, double[][] newMeans) {
        boolean different = false;
        for (int i= 0; i< means.length; i++) {
            if (getDistance(means[i], newMeans[i]) > 2) {
                different = true;
            }
        }

        return different;
    }

    public static DataC[] getCluster(DataC[] dataSet, int clusterID) {
        List<DataC> list = new ArrayList<>();
        for (DataC d: dataSet) {
            if ((int)(d.getClusterID()) == clusterID) {
                list.add(d);
            }
        }

        DataC[] cluster = new DataC[list.size()];
        list.toArray(cluster);

        return cluster;
    }

}
