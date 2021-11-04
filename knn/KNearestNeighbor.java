package project_2;

import java.util.*;

/*
    1. Load data
    2. Choose a K
    3. For each example e in the data, for query q:
        a. Call getDistance(e, q)
        b. Store the distance and id of example in an array
    4. Sort the array by distance --> O(nlog(n))
    5. Take the K smallest elements
    6.
        a. Return majority class OR weighted vote as the classification of q
        b. Return 1/d weighted average of nearest neighbors for regression
           or use Gaussian kernel

    OR

    3b. Put the distance and id of example in k-dimensional tree
    4. Find nearest neighbors in O(log(n)) time

 */
public class KNearestNeighbor {

    public DataC[] data;
    public DataC[] reducedData;
    public boolean isReducedData = false;
    public int numClasses;
    public int k;
    public double epsilon;
    public double sigma;
    public boolean classification;
   
    /*
        Constructor takes any of the data sets we are working with
     */
    public KNearestNeighbor(DataC[] dataIn, int num, boolean classification) {
    	this.data = dataIn;
    	this.numClasses = num;
        this.classification = classification;

        //some initializations for hyperparameters
        this.k = 4;
        this.epsilon = 1.0;
        this.sigma = 1.0;
    }

    /*
        Gets euclidean distance between 2 feature vectors x and y
     */
    public double getDistance(double[] x, double[] y) {
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

    /*
        Gets root mean squared error for x = predicted values, y = actual values
        Bigger penalty for bigger error compared to MAE
     */
    public double RMSE(double[] x, double[] y) {
        if (x.length != y.length) {
            return -1;
        }

        double sum = 0;
        for (int i= 0; i< x.length; i++) {
            sum += Math.pow((x[i] - y[i]), 2);
        }

        sum /= x.length;

        return Math.pow(sum, 0.5);
    }

    /*
        Gives mean absolute error where x = predicted values, y = actual values
        Basically, how far off a predicted value usually is from the actual value
     */
    public double MAE(double[] x, double[] y) {
        if (x.length != y.length) {
            return -1;
        }

        double sum = 0;
        for (int i= 0; i< x.length; i++) {
            sum += Math.abs(x[i] - y[i]);
        }

        return sum / x.length;
    }

    /*
        Corresponds to Gaussian kernel function for getting weights
        for regression
     */
    public double kernelFunc(double[] x, double[] y) {
        if (x.length != y.length) {
            return -1;
        }

        //get mean
        double xsum = 0, ysum = 0;
        for (int i= 0; i< x.length; i++) {
            xsum += x[i];
            ysum += y[i];
        }

        double mean = (xsum + ysum) / 2;
        mean = Math.pow(mean, 2);
        double sigma = 2 * Math.pow(this.sigma, 2);

        return Math.exp(-(mean / sigma));
    }

    /*
        Takes a DataC[] training set and a double[] query of features that
        make up a DataC object. Classifies the query point to a class for
        classification or a response value for regression.

        You can choose (see comments below)
            Majority voting OR weighted voting for classification
            Weighted average OR Gaussian kernel for regression
     */
    public String classify(DataC[] dataSet, double[] query) {
        DataC[] nearestNeighbors = new DataC[k];
        double[][] distArray = new double[dataSet.length][2];

        //fill distArray[i][0] w/ distance between query point and point i
        //fill distArray[i][1] w/ id of point i
        double dist;
        int i= 0;
        for (DataC d: dataSet) {
            if (d != null) {
                dist = getDistance(d.getFeatures(), query);
                distArray[i][0] = dist;
                distArray[i++][1] = d.getID();
            }
        }

        //sort distances using this to sort 2D array
        //https://stackoverflow.com/questions/4907683/sort-a-two-dimensional-array-based-on-one-column
        Arrays.sort(distArray, new Comparator<double[]>(){
            @Override
            public int compare(double[] o1, double[] o2) {
                Double dist1 = o1[0];
                Double dist2 = o2[0];
                return dist1.compareTo(dist2);
            }
        });

        //fill nearestNeighbors[] and nearestDist[]
        double[] nearestDist = new double[k];
        int j= 0, n= 0;
        while (n < nearestNeighbors.length) {
            if (distArray[j][1] > 0) {
                nearestDist[n] = distArray[j][1];
                nearestNeighbors[n++] = data[(int) distArray[j][1]-1];
            }
            j++;
        }

        String cl;

        //CLASSIFICATION
        if (this.classification) {

            //WEIGHTED VOTING
            double[] weightedVote = new double[numClasses + 2];
            int max = -1;
            int k = 1;
            for (DataC nearestNeighbor : nearestNeighbors) {
                String c = nearestNeighbor.getClassLabel();
                int classLabel = Integer.parseInt(c);
                weightedVote[classLabel] += ((double) 1 / k++);
                if (weightedVote[classLabel] > max) {
                    max = classLabel;
                }
            }

            //OR ---> MAJORITY VOTING
            //UNCOMMENT BELOW, COMMENT ABOVE FOR MAJORITY VOTING

            //take majority of nearest neighbor classes as max
            //w/ int vote[] where it increments vote[classNo]
//            int[] vote = new int[numClasses+2];
//            int max = -1;
//            for (DataC nearestNeighbor : nearestNeighbors) {
//                int classLabel = Integer.parseInt(nearestNeighbor.getClassLabel());
//                vote[classLabel]++;
//                if (vote[classLabel] > max) {
//                    max = classLabel;
//                }
//            }

            cl = String.valueOf(max);

        }

        //REGRESSION
        else {

            //WEIGHTED AVERAGE
            double weightedAvg= 0;
            double weightSum= 0;
            int c= 0;
            for (DataC d: nearestNeighbors) {
                double responseVar = Double.parseDouble(d.getClassLabel());
                weightedAvg += (responseVar / nearestDist[c]);
                weightSum += (1 / nearestDist[c++]);
            }

            weightedAvg /= weightSum;


            //OR --> GAUSSIAN KERNEL
            //UNCOMMENT BELOW, COMMENT ABOVE FOR GAUSSIAN KERNEL

//            double weightedAvg = 0;
//            double weightSum = 0;
//            int c = 0;
//            for (DataC d: nearestNeighbors) {
//                double responseVar = Double.parseDouble(d.getClassLabel());
//                weightedAvg += (responseVar * kernelFunc(d.getFeatures(), query));
//                weightSum += kernelFunc(d.getFeatures(), query);
//            }
//
//            weightedAvg /= weightSum;

            cl = String.valueOf(weightedAvg);
        }

        return cl;
    }

    /*
        L O S S --> performance of this KNN object

        Makes a confusion matrix (FP / FN are in context of c1) for classification
        ----trueClass----
               c1  ..  cN
      |    c1  TP  FP  FP
    guess  ..  FN  TP
      |    cN  FN      TP

        For regression, accuracy, root mean squared error, and mean absolute error
        are calculated

        Returns performance[] = 3 loss stats
            Precision, recall, F1 for classification
            RSME, MAE, accuracy for regression

     */
    public double[] loss(DataC[] dataSet, DataC[] testSet) {

        int[][] confusionMatrix = new int[numClasses][numClasses];
        double[] performance = new double[3];
        int count= 0;

        //CLASSIFICATION
        if (this.classification) {
            for (DataC d : testSet) {
                if (d != null) {
                    double[] test = d.getFeatures();
                    String trueClass = d.getClassLabel();
                    String guess = classify(dataSet, test);
                    int guessNum = Integer.parseInt(guess) - 1;
                    int trueNum = Integer.parseInt(trueClass) - 1;

                    confusionMatrix[guessNum][trueNum]++;
                    if (guessNum == trueNum) {
                        count++;
                    }

                }
            }

            //get precision and recall
            double precision;
            int TP, TPsum= 0;
            int FP, FPsum= 0;
            int i= 0;
            for (int[] row: confusionMatrix) {
                FP = 0;
                TP = row[i];

                for (int j= 1; j< numClasses; j++) {
                    FP += row[(i+j)%numClasses];
                }

                TPsum += TP;
                FPsum += (FP + TP);
                i++;
            }

            precision = (double)TPsum / FPsum;

            double recall;
            TPsum = 0;
            int FN, FNsum = 0;
            for (int j= 0; j< numClasses; j++) {
                FN = 0;
                TP = confusionMatrix[j][j];

                for (int k= 1; k< numClasses; k++) {
                    FN += confusionMatrix[(j+k)%numClasses][j];
                }

                TPsum += TP;
                FNsum += (FN + TP);
            }

            recall = (double)TPsum / FNsum;

            //F1 score - harmonic mean of precision and recall
            double F1 = (2 * precision * recall) / (precision + recall);

            performance[0] = precision;
            performance[1] = recall;
            performance[2] = F1;
        }

        //REGRESSION
        else {
            double[] predicted = new double[testSet.length];
            double[] actual = new double[testSet.length];
            int c= 0;

            for (DataC d: testSet) {
                if (d != null) {
                    double[] test = d.getFeatures();
                    double trueResponseVar = Double.parseDouble(d.getClassLabel());
                    double guess = Double.parseDouble(classify(dataSet, test));

                    predicted[c] = guess;
                    actual[c++] = trueResponseVar;

                    if (Math.abs(trueResponseVar - guess) < this.epsilon) {
                        count++;
                    }
                }
            }

            performance[0] = RMSE(predicted, actual);
            performance[1] = MAE(predicted, actual);
            performance[2] = (double)count / testSet.length;
        }

//        for(int[] row: confusionMatrix) {
//            System.out.println(Arrays.toString(row));
//        }
//        System.out.println("---------------------");


//        System.out.println(count + " / " + testSet.length);
//        double accuracy = (double) count / testSet.length;
//        System.out.println(accuracy);

        return performance;
    }

    /*
        Test performance by 10-fold cross-validation

        Returns an array of averaged loss stats (see loss() for metrics)
     */
    public double[] crossValidate() {
        int trainLen = (int)(0.9 * data.length) + 1;

        if (this.isReducedData) {
            trainLen += this.reducedData.length;
        }

        int testLen = data.length - trainLen;
        System.out.println("Train: " + trainLen);
        System.out.println("Test:  " + testLen);
        DataC[] train = new DataC[trainLen];
        DataC[] test;
        DataC[][] split = new DataC[10][testLen];
        double[] lossStats = new double[3];

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

            //train fold to (fold+8)%10, test on (fold+9)%10
            List<DataC> list = new ArrayList<>();
            for (int i= 0; i< 9; i++) {
                //add 1/10 of data to list 9 times
                list.addAll(Arrays.asList(split[(i+fold)%10]));
            }

            if (this.isReducedData) {
                list.addAll(Arrays.asList(this.reducedData));
            }

            //fill train[] and test[]
            list.toArray(train);
            test = split[(fold+9)%10];

            //get loss data
            int i = 0;
            double[] lossTemp = loss(train, test);
            for (double d: lossTemp) {
                lossStats[i++] += d;
            }
        }

        //average loss data
        int l = 0;
        for (double d: lossStats) {
            lossStats[l++] = d / 10;
        }

        //print lossStats
        printLossStats(lossStats);

        return lossStats;
    }

    /*
        A little helper function to print loss
        metrics nicely.
     */
    public void printLossStats(double[] stats) {
        if (this.classification) {
            System.out.println("Precision: "+ stats[0]);
            System.out.println("Recall:    "+ stats[1]);
            System.out.println("F1 score:  "+ stats[2]);
        }
        else {
            System.out.println("RSME:     "+ stats[0]);
            System.out.println("MAE:      "+ stats[1]);
            System.out.println("Accuracy: "+ stats[2]);
        }
    }

    /*
        Tune k, epsilon, sigma by random search.
            Evaluate 10 different sets of hyperparameters
            and choose the best.

        RANGES OF VALUES - defined by me
            k -> {1, 2, ..., 20}
            epsilon -> [0.3, 3]
            sigma -> [0.3, 3]
     */
    public void tune() {
        double[] bestParams = new double[3];
        bestParams[0] = (int) (Math.random() * 20) + 1;     //initialize k
        bestParams[1] = (Math.random() * 3) + 0.3;          //initialize epsilon
        bestParams[2] = (Math.random() * 3) + 0.3;          //initialize sigma
        this.k = (int)bestParams[0];
        this.epsilon = bestParams[1];
        this.sigma = bestParams[2];

        double lossMetricBest;
        if (this.classification) {
            //use F1 for classification
            lossMetricBest = crossValidate()[2];
        }
        else {
            //use RSME for regression
            lossMetricBest = crossValidate()[0];
        }

        for (int i= 0; i< 10; i++) {
            double[] newParams = new double[3];
            newParams[0] = (int) (Math.random() * 20) + 1;
            newParams[1] = (Math.random() * 3) + 0.3;
            newParams[2] = (Math.random() * 3) + 0.3;

            this.k = (int) newParams[0];
            this.epsilon = newParams[1];
            this.sigma = newParams[2];

            double lossMetric;
            if (this.classification) {
                //use F1 for classification
                lossMetric = crossValidate()[2];
                if (lossMetric > lossMetricBest) {
                    bestParams = newParams;
                    lossMetricBest = lossMetric;
                }
            }
            else {
                //use RSME for regression
                lossMetric = crossValidate()[0];
                if (lossMetric < lossMetricBest) {
                    bestParams = newParams;
                    lossMetricBest = lossMetric;
                }
            }

        }

        System.out.println("Best k:       "+ bestParams[0]);
        System.out.println("Best epsilon: "+ bestParams[1]);
        System.out.println("Best sigma:   "+ bestParams[2]);
        if (this.classification) {
            System.out.println("F1 score:     " + lossMetricBest);
        }
        else {
            System.out.println("RSME:         " + lossMetricBest);
        }

        this.k = (int) bestParams[0];
        this.epsilon = bestParams[1];
        this.sigma = bestParams[2];
    }

    /*
        EDITED KNN
        Continue removing bad / useless samples that classify incorrectly
        until no more are removed.
     */
    public void editDataSet() {
        List<DataC> editedSet = Arrays.asList(this.data);
        boolean dataDeleted;

        do {
            DataC[] editedSetArr = new DataC[editedSet.size()];
            editedSet.toArray(editedSetArr);
            System.out.println(editedSet.size());

            int i = 0;
            dataDeleted = false;
            for (DataC d : editedSet) {
                double trueClass = Double.parseDouble(d.getClassLabel());
                double guess = Double.parseDouble(classify(editedSetArr, d.getFeatures()));

                if (this.classification) {
                    if (guess != trueClass) {
                        editedSetArr[i] = null;
                        dataDeleted = true;
                    }
                } else {
                    if (Math.abs(trueClass - guess) > this.epsilon) {
                        editedSetArr[i] = null;
                        dataDeleted = true;
                    }
                }
                i++;
            }

            List<DataC> list = new ArrayList<>();
            for (DataC d : editedSetArr) {
                if (d != null) {
                    list.add(d);
                }
            }

            editedSet = list;
        } while (dataDeleted);

        this.data = new DataC[editedSet.size()];
        editedSet.toArray(this.data);

        //need to set new id numbers to avoid array out of bounds error
        int i= 1;
        for (DataC d: this.data) {
            d.setID(i++);
        }
    }

    /*
        CONDENSED KNN
        Select the smallest subset Z of this.data such that
        using Z in place of this.data does not degrade performance
     */
    public void condenseDataSet() {
        List<DataC> condensedSet = new ArrayList<>();
        DataC[] condensedArr = this.data;
        int rand = (int)(Math.random()*this.data.length);
        condensedSet.add(this.data[rand]);
        condensedArr[rand] = null;

        boolean changed = true;

        //REPEAT while samples are being added to condensedSet
        while(changed && condensedSet.size() < this.data.length) {
            changed = false;

            //shuffle condensed arr
            List<DataC> temp = Arrays.asList(condensedArr);
            Collections.shuffle(temp);
            DataC[] shuffled = new DataC[data.length];
            temp.toArray(shuffled);
            condensedArr = shuffled;

            //for each d in the data set, find d' in Z s.t. it is the closest neighbor of d
            //if class(d) != class(d') add d to condensedSet
            for (DataC d: condensedArr) {
                double minDist = Double.POSITIVE_INFINITY;
                DataC min = null;
                double dist;

                if (d != null) {
                    for (DataC x : condensedSet) {
                        dist = getDistance(d.getFeatures(), x.getFeatures());

                        if (dist < minDist) {
                            minDist = dist;
                            min = x;
                        }
                    }
                }

                if (min != null) {
                    if (!d.getClassLabel().contentEquals(min.getClassLabel())) {
                        condensedSet.add(d);
                        changed = true;
                    }
                    d = null;
                }

            }
//            System.out.println("Size of condensed: "+ condensedSet.size());

        }

        this.data = new DataC[condensedSet.size()];
        condensedSet.toArray(this.data);

        //need to set new id numbers to avoid array out of bounds error
        int i= 1;
        for (DataC d: this.data) {
            d.setID(i++);
        }
    }

    /*
        Augment the data set with k cluster centroids
        using k-means clustering
     */
    public void kMeansClusters(int k) {
        DataC[] centroids = Cluster.kMeansClusters(this.data, k);

        this.reducedData = new DataC[centroids.length];
        this.reducedData = centroids;
        this.isReducedData = true;

        //need to set new id numbers to avoid array out of bounds error
        int i= 1;
        for (DataC d: this.reducedData) {
            d.setID(i++);
        }
    }

    /*
        Augment the data set with k cluster medoids
        using k-medoids clustering
     */
    public void kMedoidsClusters(int k) {
        DataC[] centroids = Cluster.kMedoidsClusters(this.data, k);

        this.reducedData = new DataC[centroids.length];
        this.reducedData = centroids;
        this.isReducedData = true;

        //need to set new id numbers to avoid array out of bounds error
        int i= 1;
        for (DataC d: this.reducedData) {
            d.setID(i++);
        }
    }

    /*
        Uncomment the one(s) you want to look at (I would recommend 1 at a time).
        Choose ONE of the following
            1. Regular KNN by commenting out all of the calls below
            2. Edited KNN by uncommenting KNN.editDataSet
            3. Condensed KNN by uncommenting KNN.condenseDataSet
            4. k-Means augmented KNN by uncommenting KNN.kMeansClusters
            5. k-Medoids augmented KNN by uncommenting KNN.kMedoidsClusters
                For the last 2, it is recommended that the argument be:
                    KNN.numClasses for classification
                    (int) Math.sqrt(KNN.data.length) for regression

        Leave tune() uncommented to see performance.
     */
    public static void main(String[] args) {

        //-------------------------------------------GLASS
        System.out.println("GLASS-------------------------");
        String gdFileName = "data-sets/glass.data";
        DataSetUp gdSetUp = new DataSetUp(gdFileName, "end","classification", false);

        KNearestNeighbor glassKNN = new KNearestNeighbor(gdSetUp.getAllData(), 7, true);

//        glassKNN.editDataSet();       //<-- BEST BY FAR
//        glassKNN.condenseDataSet();   //<--ALL STATS ARE BAD W/ GLASS (slightly better)
//        glassKNN.kMeansClusters(glassKNN.numClasses); //<-- ONLY ONE CLUSTER
//        glassKNN.kMedoidsClusters(glassKNN.numClasses);   //<-- better

//        glassKNN.tune();

        //-------------------------------------------HOUSE VOTES
        System.out.println("HOUSE VOTES-------------------------");
        String hvdFileName = "data-sets/house-votes-84.data";
        DataSetUp hvdSetUp = new DataSetUp(hvdFileName, "beg","classification", false);

        KNearestNeighbor houseKNN = new KNearestNeighbor(hvdSetUp.getAllData(), 2, true);

//        houseKNN.editDataSet();                           //<-- ?
//        houseKNN.condenseDataSet();                       //<---AMAZING sometimes?
//        houseKNN.kMeansClusters(houseKNN.numClasses);     //<-- better
//        houseKNN.kMedoidsClusters(houseKNN.numClasses);   //<-- better

//        houseKNN.tune();

        //-------------------------------------------SEGMENTATION
        System.out.println("SEGMENTATION-------------------------");
        String sdFileName = "data-sets/segmentation.data";
        DataSetUp sdSetUp = new DataSetUp(sdFileName, "beg","classification", false);

        KNearestNeighbor segmentationKNN = new KNearestNeighbor(sdSetUp.getAllData(), 7, true);

//        segmentationKNN.editDataSet();                                    //<-- better
//        segmentationKNN.condenseDataSet();                                //<-- better
//        segmentationKNN.kMeansClusters(segmentationKNN.numClasses);       //<-- slightly better
//        segmentationKNN.kMedoidsClusters(segmentationKNN.numClasses);     //<-- better

//        segmentationKNN.tune();

        //-------------------------------------------ABALONE
        System.out.println("ABALONE-------------------------");
        String adFileName = "data-sets/abalone.data";
        DataSetUp adSetUp = new DataSetUp(adFileName, "endA","regression", false);

        KNearestNeighbor abaloneKNN = new KNearestNeighbor(adSetUp.getAllData(), 0, false);

//		abaloneKNN.editDataSet();                                                   //<-- BEST
//		abaloneKNN.condenseDataSet();                                               //<-- slightly worse
//        abaloneKNN.kMeansClusters((int) Math.sqrt(abaloneKNN.data.length));       //<-- same
//        abaloneKNN.kMedoidsClusters((int) Math.sqrt(abaloneKNN.data.length));     //<-- slightly better

//        abaloneKNN.tune();

        //------------------------------------------FOREST FIRES
        System.out.println("FOREST FIRES-------------------------");
        String ffdFileName = "data-sets/forestfires.data";
        DataSetUp ffdSetUp = new DataSetUp(ffdFileName, "endF","regression", false);

        KNearestNeighbor forestFireKNN = new KNearestNeighbor(ffdSetUp.getAllData(), 0, false);

//        forestFireKNN.editDataSet();                                                    //<-- REALLY GOOD PERFORMANCE
//        forestFireKNN.condenseDataSet();                                                //<-- slightly worse
//        forestFireKNN.kMeansClusters((int) Math.sqrt(forestFireKNN.data.length));       //<-- slightly worse
//        forestFireKNN.kMedoidsClusters((int) Math.sqrt(forestFireKNN.data.length));     //<-- worse

//        forestFireKNN.tune();

        //------------------------------------------MACHINE
        System.out.println("MACHINE-------------------------");
        String mdFileName = "data-sets/machine.data";
        DataSetUp mdSetUp = new DataSetUp(mdFileName, "endM","regression", true);

        KNearestNeighbor machineKNN = new KNearestNeighbor(mdSetUp.getAllData(), 0, false);

		//MACHINE IS NOT WORKING
//		machineKNN.editDataSet();       //<-- does not work
//		machineKNN.condenseDataSet();   //<-- better?
//        machineKNN.kMeansClusters((int) Math.sqrt(machineKNN.data.length));     //<-- worse
//        machineKNN.kMedoidsClusters((int) Math.sqrt(machineKNN.data.length));   //<-- better?

//        machineKNN.tune();

    }
}
