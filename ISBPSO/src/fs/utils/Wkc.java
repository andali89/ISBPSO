/**
 * Copyright (c) 2024, An-Da Li 
 * Coded by An-Da Li
 * email: andali1989@163.com
 *
 * This is an implementation of Improved Binary Sticky PSO (ISBPSO) for Feature Selection. 
 * For a detailed description of the method please refer to 
 *
 * Li, A.-D.*, Xue, B., & Zhang, M. (2021). Improved binary particle swarm optimization for 
 * feature selection with new initialization and search space reduction strategies. 
 * Applied Soft Computing, 106, 107302.
 *
 */

package fs.utils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.*;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.classifiers.functions.LibSVM;

/**
 * @author lad
 *
 */
public class Wkc {
	public AbstractClassifier classifier;
	public AbstractClassifier spareclassifier = new IBk();
	public Instances traindata;
	public Instances testdata;
	public Instances data;
	public int fold;
	public String classiname = "";
	public boolean useSpareClassi = false;
	public Wkc(String classifiername) throws Exception {
	// initialize a classifier according to classifier name.
		this.classiname = classifiername;
		if(classifiername.equals("NB") ) {
			this.classifier = new NaiveBayes();			
		}
		if(classifiername.equals("KNN")) {
			this.classifier = new IBk();			
			this.classifier.setOptions(new String[] {"-K", "1"});			
		}if(classifiername.equals("SVM")) {
			this.classifier = new LibSVM();
		}		
	}
	public Wkc(AbstractClassifier classifier) throws Exception {
		// initialize a classifier according to classifier name.
			this.classifier = (AbstractClassifier) AbstractClassifier.makeCopy((Classifier)classifier);	
		}
	public void setTtdata(Instances traindata, Instances testdata,double[] fs) {
		//set the traindata and testdata
		try {
			this.traindata = Basicfunc.select(traindata, fs);
			this.testdata = Basicfunc.select(testdata, fs);	
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			
	}
	
	
	/**
	 * set the traindata and testdata
	 * @param traindata
	 * @param testdata
	 */
	public void setTtdata(Instances traindata, Instances testdata) {
		
		this.traindata = new Instances(traindata);
		this.testdata = new Instances(testdata);
	}
	
	public double[] run() throws Exception {
		// return the classification accuracy on each class and overall classification accuracy.
		this.classifier.buildClassifier(this.traindata);
		double[] acc = predic(this.testdata);
		if(this.useSpareClassi) {
			this.spareclassifier.buildClassifier(this.traindata);
			double[] spareAcc = predic(this.testdata, this.spareclassifier);
			for(int i = 0; i < acc.length; i++) {
				acc[i] = (acc[i] + spareAcc[i]) / 2;
			}
		}
		
		
		return acc;
		
		
	}
	public double[] getTrainacc() throws Exception {
		// note that this function must be used after the run conducted (i.e. the classifier has been built)
		this.classifier.buildClassifier(this.traindata);
		return predic(this.traindata);
	}
	private double[] predic(Instances testdata, AbstractClassifier classifier) throws Exception {
		double[][] prob = classifier.distributionsForInstances(testdata);
		double[] predic = Matcd.maxMatrix(prob, 0)[0];
		double[] actual = testdata.attributeToDoubleArray(testdata.classIndex());
		int numclass = prob[0].length;
		double[] acc = new double[numclass + 1]; //TP,TN,ACC
		double[] posnegNum = new double[numclass] ;
		for(int i = 0; i < numclass; i++) {
			acc[i] = 0;
			posnegNum[i]  = 0;
		}
		acc[numclass] = 0;
		for(int i = 0; i < actual.length; i++) {
			for(int j = 0; j < numclass; j++) {
				if(actual[i] == j) {
					posnegNum[j]++;
					if(actual[i] == predic[i]) {
						acc[j]++;
						acc[numclass]++;
					}
				}
			}			
		}
		for(int i = 0; i < numclass; i++) {
			acc[i] = acc[i]/posnegNum[i];			
		}
		acc[numclass] = acc[numclass]/actual.length;
		return acc;
		
	}
	private double[] predic(Instances testdata) throws Exception {
		return predic(testdata, this.classifier);
		
	}
	
	public void setData(Instances data, int fold) {
		//seed == -1 random sorting the data
		this.data = data;
		this.fold = fold;		
	}
	
	public double[] cvRun() throws Exception {
		//return average results of K folds
		int numclass = this.data.numClasses();
		int numInstances = this.data.numInstances();
		double[] acc = new double[numclass + 1];
		for(int i = 0; i < acc.length; i++) {
			acc[i] = 0;
			
		}
		double[] temp = new double[numclass + 1];
		for(int i = 0; i < this.fold; i++) {
			Instances testcv = this.data.testCV(this.fold, i);
			int numTest = testcv.numInstances();
			this.setTtdata(this.data.trainCV(this.fold, i), testcv);
			temp = this.run();
			
			for(int j = 0; j < acc.length; j++) {
				
				acc[j] += (double) temp[j] *(double) numTest /(double) numInstances; 
			}
		}
		
		return acc;
	}
	 


}
