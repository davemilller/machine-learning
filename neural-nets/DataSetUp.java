import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/*
	A class to handle high level data set up

	@Author: Dave Miller
 */
public class DataSetUp {
	public DataC[] data;
	public final int length;
	public boolean isClassification;
    
	public DataSetUp(String fileName, String classPos, String clOrReg) {
		
		BufferedReader reader; //creates a buffered reader
        DataC data;
        
        List<DataC> list = new ArrayList<>();

        try { //Reads in the file and checks for exception
            reader = new BufferedReader(new FileReader(fileName));
            //reads the file in line by line
            String line = reader.readLine();
            int lineNo = 0;
            //While we are not at the end of the file do things in the while loop
            while(line != null) {
                data = new DataC(line, lineNo+1, classPos, clOrReg);
                list.add(data);

                line = reader.readLine();
            }

        }catch (IOException e) {
            e.printStackTrace();
        }

        this.data = new DataC[list.size()];
        this.length = list.size();
		list.toArray(this.data);

		this.isClassification = clOrReg.contentEquals("classification");
	        
	}

	public DataC[] getAllData() {
		return this.data;
	}

	public void zScoreNormalize() {
		//get means of all features
		//get SD of all features
		//change features
		double[] means = new double[data[0].getFeatures().length];
		for (DataC d: data) {
			double[] features = d.getFeatures();

			for (int i= 0; i< features.length; i++) {
				means[i] += features[i];
			}
		}

		for (int i= 0; i< means.length; i++) {
			means[i] /= length;
		}

		//get SDs
		double[] SDs = new double[data[0].getFeatures().length];
		for (DataC d: data) {
			double[] features = d.getFeatures();

			for (int i= 0; i< features.length; i++) {
				SDs[i] += Math.pow((features[i] - means[i]), 2);
			}
		}

		for (int i= 0; i< SDs.length; i++) {
			SDs[i] = Math.pow((SDs[i] / length), 0.5);
		}

		//normalize data
		for (DataC d : data) {
			double[] normFeatures = new double[data[0].getFeatures().length];
			double[] features = d.getFeatures();

			for (int j = 0; j < features.length; j++) {
				normFeatures[j] = (features[j] - means[j]) / (SDs[j] + 0.01);
			}
			d.setNormFeatures(normFeatures);
		}

		//if regression, scale response value to [0, 1]
		if (!isClassification) {
			double max = Double.NEGATIVE_INFINITY;
			double min = Double.POSITIVE_INFINITY;
			for (DataC d: data) {
				double r = Double.parseDouble(d.getClassLabel());
				if (r > max) { max = r; }
				if (r < min) { min = r; }
			}
			for (DataC d: data) {
				double r = Double.parseDouble(d.getClassLabel());
				d.setClassLabel(String.valueOf( (r - min) / (max - min) ));
			}
		}
	}

}
