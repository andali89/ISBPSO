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

package fs;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.text.DecimalFormat;

import fs.pso.PSO;
import fs.pso.Maskbpso;
import fs.eval.ModiWrapperSubsetEval;
import fs.utils.Basicfunc;
import fs.utils.Matcd;
import fs.utils.Wkc;
import weka.attributeSelection.ASEvaluation;
import weka.core.Instances;

public class RunPSO {
	/**
	 * set parameters
	 */
	static int popNum = 30;
	static int iterTime = 100;
	static int stickNum = 50;
	static double[] impg = {0.25,0.25,0.5};
	//static double[] impg = {0.33,0.33,0.34};
	static int[] maskset = {20,40,60,80};
	static String classifierName = "KNN";
	static int knnNum = 5;
	static int objtype = 0; //0 denotes the acc and number
	//int retime = retimein; // number of repeating
	static int seed = 1;
	static double elimpercent = 0;
	static int innerfold = 5;
	static double[] sigPara = new double[] {0, 0.8, 10};
	static boolean ifnorm = true; // using standardization
	

	
	static int step = 20;
	static int retime = 30;  //repetition time 

	public static void main(String[] args) throws Exception {
		// run FSIPSO
		
		weka.core.Instances data = null;
		int foldop = -1; // when reading the data do not stratify
		String filpath = "./data/wbcd/wbcd.arff";
		data = Basicfunc.readData(filpath, seed, ifnorm, foldop);

		// divide the data into trainset and testset where 30% percent is the testset
		
		Instances[] data2 = Basicfunc.divData(data, 3, 10, innerfold);

		/**
		 * initialize the PSOs
		 */

		Wkc wkc = new Wkc(classifierName);
		if (classifierName.equals("KNN")) {
			wkc.classifier.setOptions(new String[] { "-K", String.valueOf(knnNum) });
		}

		// define and set evaluator
		ASEvaluation subsetEval;
		subsetEval = new ModiWrapperSubsetEval();
		((ModiWrapperSubsetEval) subsetEval).setClassifier(wkc.classifier);

		PSO pso = null;
		pso = new Maskbpso(data2[0], wkc, popNum, iterTime, stickNum, impg, objtype, maskset) ;
  	   ((Maskbpso) pso).setElimPercent(elimpercent);
  	   

  	   
  	  ((Maskbpso) pso).setiniMethod("hybrid");
  	  ((Maskbpso) pso).setGAop(true, 5);    		   
  	  
  	   ((Maskbpso) pso).sigPara = sigPara;
  	 
  	   

		subsetEval.buildEvaluator(data2[0]);
		pso.setEvaluator(subsetEval);

		

		long rnd = 1;
		double[][] re = new double[retime][7];
		double[][] info = new double[retime][iterTime + 1];

		for (int i = 0; i < retime; i++) {
			pso.setSeed(rnd);
			pso.run(); // get the solutions
			getTestResult(i, re, info, pso, data2[0], data2[1], rnd);
		}

		// Output info

		try {
			String outPath = "re.csv"; // output name;
			// File file = new File(outPath);
			OutputStream fop = new FileOutputStream(outPath);
			OutputStreamWriter writer = new OutputStreamWriter(fop, "UTF-8");
			writeInfo("data", writer, re, info);
			writer.flush();
			writer.close();
			fop.close();

		} catch (Exception e) {
			// TODO: handle exception
			System.out.println(e);
		}

	}

	public static void getTestResult(int i, double[][] re, double[][] iterInfo, PSO pso, Instances trainset,
			Instances testset, long rnd) throws Exception {

		Wkc classifier = new Wkc(new weka.classifiers.lazy.IBk(knnNum));		
		classifier.setTtdata(trainset, testset, pso.gbest());
		double[] tempacc = classifier.run();
		re[i][0] = (int) i + 1.0;
		re[i][1] = tempacc[tempacc.length - 1];
		// train accuracy on training data
		tempacc = classifier.getTrainacc();
		re[i][2] = tempacc[tempacc.length - 1];

		re[i][3] = pso.getNum(pso.gbest());
		re[i][4] = pso.gbestFit();
		re[i][5] = pso.runtime();
		re[i][6] = (double) rnd;
		iterInfo[i] = Matcd.mergeVector(new double[] { i + 1 }, pso.iterInfo());
		System.out.printf("Testing Accuray=%f, Training Accuracy=%f, numberSelected=%f\r\n", re[i][1], re[i][2],
				re[i][3]);
	}

	public static void writeInfo(String methodname, OutputStreamWriter writer, double[][] result, double[][] iterInfo)
			throws IOException, Exception {
		/**
		 * write the results of PSO
		 */
		writer.append(methodname + "\r\n");
		writer.append("repeat, testacc, trainacc, featurenum, fitness,runtime, rndseed\r\n");
		writefile(result, writer);
		writer.append("Avg.,");
		writefile(Matcd.meanMatrix(result, 1), writer, true);
		writer.append("Std.,");
		writefile(Matcd.stdMatrix(result, 0, 1), writer, true);
		writer.append("\r\n\r\n");
		/**
		 * iteration info
		 */
		// PSO
		writer.append("iteration info of " + methodname + "\r\n");
		writefile(iterInfo, writer);
		writer.append("iterAvg.,");
		writefile(Matcd.meanMatrix(iterInfo, 1), writer, true);
		writer.append("\r\n\r\n");
		writer.flush();

	}
	
	@SuppressWarnings("unused")
	private static void writefile(double[] data, OutputStreamWriter writer) throws IOException {
		// write a row to the file
		int J = data.length;
		DecimalFormat df = new DecimalFormat("#.############");
		String value;

		for (int j = 0; j < J - 1; j++) {
			value = df.format(data[j]);
			writer.append(value + ",");
		}
		value = df.format(data[J - 1]);
		writer.append(value + "\r\n");
		writer.flush();

	}

	@SuppressWarnings("unused")
	private static void writefile(int[] data, OutputStreamWriter writer) throws IOException {
		// write a row to the file
		int J = data.length;
		DecimalFormat df = new DecimalFormat("#.############");
		String value;

		for (int j = 0; j < J - 1; j++) {
			value = df.format(data[j]);
			writer.append(value + ",");
		}
		value = df.format(data[J - 1]);
		writer.append(value + "\r\n");
		writer.flush();

	}
	
	private static void writefile(double[] data, OutputStreamWriter writer, boolean first) throws IOException {
		// write a row to the file
		// if first == true then do not write the first one;
		int J = data.length;
		DecimalFormat df = new DecimalFormat("#.############");
		String value;
		int start = 0;
		if (first = true) {
			start = 1;
		}
		for (int j = start; j < J - 1; j++) {
			value = df.format(data[j]);
			writer.append(value + ",");
		}
		value = df.format(data[J - 1]);
		writer.append(value + "\r\n");
		writer.flush();

	}
	
	private static void writefile(double[][] data, OutputStreamWriter writer) throws IOException {
		int I = data.length;
		int J = data[0].length;
		DecimalFormat df = new DecimalFormat("#.############");
		String value;

		for (int i = 0; i < I; i++) {
			for (int j = 0; j < J - 1; j++) {
				value = df.format(data[i][j]);
				writer.append(value + ",");

			}
			value = df.format(data[i][J - 1]);
			writer.append(value + "\r\n");
			writer.flush();
		}

	}

}
