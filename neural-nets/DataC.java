/*
    A generalized Data object that corresponds to 1 line in a _.data file

    @Author: Dave Miller
 */
public class DataC {

    private String allData;
    private double[] features;
    private double[] normalizedFeatures;
    private String classLabel;
    private int id;

    public DataC(String fullData, int id, String classLoc, String clOrReg) { 
    	if(clOrReg.equals("classification")){
	    	if(classLoc.equals("end")) {
		        this.allData = fullData;
		        String[] tokens = fullData.split(",");
		
		        features = new double[tokens.length - 2];
		
		        for (int i= 1; i< tokens.length -1; i++) {
		            features[i-1] = Double.parseDouble(tokens[i]);
		        }
		
		        classLabel = tokens[tokens.length-1];
		        this.id = id;
	    	} else if (classLoc.contentEquals("endS")) {		//soybean
	    		this.allData = fullData;
	    		String[] tokens = fullData.split(",");

	    		features = new double[tokens.length - 2];

	    		for (int i= 1; i< tokens.length-1; i++) {
	    			features[i-1] = Double.parseDouble(tokens[i]);
				}

	    		classLabel = tokens[tokens.length-1].split("D")[1];
			} else if (classLoc.contentEquals("endB")) {		//breast cancer
	    		this.allData = fullData;
	    		String[] tokens = fullData.split(",");

	    		features = new double[tokens.length - 2];

	    		for (int i= 1; i< tokens.length-1; i++) {
	    			if (tokens[i].contentEquals("?")) {
	    				tokens[i] = String.valueOf((int)(Math.random()*5) + 1);
					}
	    			features[i-1] = Double.parseDouble(tokens[i]);
				}

	    		if (tokens[tokens.length-1].contentEquals("2")) {
	    			classLabel = "1";
				}
	    		else if (tokens[tokens.length-1].contentEquals("4")){
	    			classLabel = "2";
				}
			}
	    	else if (classLoc.equals("beg")) {
		        this.allData = fullData;
		        String[] tokens = fullData.split(",");
		
		        features = new double[tokens.length-1];
		        
		        for (int i= 1; i< tokens.length; i++) {
		            features[i-1] = Double.parseDouble(tokens[i]);
		        }
		
		        classLabel = tokens[0];
		        
		        this.id = id;
	    	}
    	}
    	if(clOrReg.equals("regression")){
    		//This is for FOREST FIRES DATA
    		if(classLoc.equals("endF")) {
			 	this.allData = fullData;
		        String[] tokens = fullData.split(",");
		
		        features = new double[tokens.length+1];
		        
		       //This is for when we read in the Regression Forest Fires Data to change categorical data into numbers
		        for(int i = 1; i < tokens.length; i++) {
					switch (tokens[i]) {
						case "jan" : tokens[i] = Integer.toString(0);
						case "feb" : tokens[i] = Integer.toString(1);
						case "mar" : tokens[i] = Integer.toString(2);
						case "apr" : tokens[i] = Integer.toString(3);
						case "may" : tokens[i] = Integer.toString(4);
						case "jun" : tokens[i] = Integer.toString(5);
						case "jul" : tokens[i] = Integer.toString(6);
						case "aug" : tokens[i] = Integer.toString(7);
						case "sep" : tokens[i] = Integer.toString(8);
						case "oct" : tokens[i] = Integer.toString(9);
						case "nov" : tokens[i] = Integer.toString(10);
						case "dec" : tokens[i] = Integer.toString(11);
					}
		        }
		        
		      //This is for when we read in the Regression Forest Fires Data to change categorical data into numbers
		        for(int i = 1; i < tokens.length; i++) {
					switch (tokens[i]) {
						case "mon" : tokens[i] = Integer.toString(0);
						case "tue" : tokens[i] = Integer.toString(1);
						case "wed" : tokens[i] = Integer.toString(2);
						case "thu" : tokens[i] = Integer.toString(3);
						case "fri" : tokens[i] = Integer.toString(4);
						case "sat" : tokens[i] = Integer.toString(5);
						case "sun" : tokens[i] = Integer.toString(6);
					}
		        }

		        int j=0;
		        for (int i= 0; i< tokens.length-1; i++) {
		        	if (i == 2) {
		        		//make month cyclical
						features[j++] = Math.sin(Double.parseDouble(tokens[i]) * (2 * Math.PI / 12));
						features[j++] = Math.cos(Double.parseDouble(tokens[i]) * (2 * Math.PI / 12));
					}
		        	else if (i == 3) {
		        		//make week cyclical
		        		features[j++] = Math.sin(Double.parseDouble(tokens[i]) * (2 * Math.PI / 7));
						features[j++] = Math.cos(Double.parseDouble(tokens[i]) * (2 * Math.PI / 7));
					}
		        	else {
						features[j++] = Double.parseDouble(tokens[i]);
					}
		        }
		
		        classLabel = tokens[tokens.length-1];
		        this.id = id;
		    //THIS IS FOR ABALONE DATA   
    		}else if(classLoc.equals("endA")) {
    			
			 	this.allData = fullData;
		        String[] tokens = fullData.split(",");
		
		        features = new double[tokens.length-1];

		        switch(tokens[0]) {
					case "M" : features[0] = 1.0;
					case "F" : features[0] = 2.0;
					case "I" : features[0] = 3.0;
				}

		        for (int i= 1; i< tokens.length-1; i++) {
		            features[i] = Double.parseDouble(tokens[i]);
		        }
		
		        classLabel = tokens[tokens.length-1];
		        this.id = id;
		    //THIS IS FOR MACHINE DATA    
    		}else {
    			
			 	this.allData = fullData;
		        String[] tokens = fullData.split(",");
		
		        features = new double[tokens.length-4];

		        for (int i= 2; i< tokens.length-2; i++) {
		            features[i-2] = Double.parseDouble(tokens[i]);
		        }
		
		        classLabel = tokens[tokens.length-2];
		        this.id = id;
    		}
    	}
    }

	public String getAllData() { return allData; }

    public double[] getFeatures() {
        return features;
    }

    public void setNormFeatures(double[] normFeatures) { this.normalizedFeatures = normFeatures; }

    public double[] getNormalizedFeatures() { return this.normalizedFeatures; }

    public String getClassLabel() {
        return classLabel;
    }

    public void setClassLabel(String cl) {
    	this.classLabel = cl;
	}

    public int getID() {
        return id;
    }

    public void setID(int id) { this.id = id; }

}
