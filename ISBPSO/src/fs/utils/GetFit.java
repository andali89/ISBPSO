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

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class GetFit {
	private AbstractClassifier classi;
	private Instances data;
	private int fNum; //full number of features
	private Objfuc obj;
	private int innerfold;
	private Evaluation m_Evaluation;
	public GetFit(Instances data,AbstractClassifier classi, Objfuc obj, int innerfold) {
		// TODO Auto-generated constructor stub
		this.data = data;
		this.classi = classi;
		this.obj = obj;
		this.innerfold = innerfold;
		this.fNum = data.numAttributes() - 1;
	}
	
	public double getFitness(double[] pos) {
		double[] acc = null;
		try {
			acc = getAcc(pos);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return getFitness(acc, this.obj);		
	}
	
	public double getFitness(double[] acc, Objfuc obj)  {
		// get the fitness of the obj, fs denote the number			
		return  obj.fit(acc, (int) acc[0], fNum);

	}
	protected double[] getAcc(double[] pos)  {
		double[] acc = new double[3];
		Random Rnd = new Random(1);
		Instances datain = null;
		try {
			datain = Basicfunc.select(this.data, pos);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		int fs = datain.numAttributes() -1;		
		try {
			m_Evaluation = new Evaluation(datain);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		if(fs > 0) {
					
			acc[0] = fs;
			acc[1] =0.5;
			try {
				m_Evaluation.crossValidateModel((Classifier)this.classi, datain, innerfold,
						Rnd);
				//acc = classi.cvRun();
				acc[2] = 1 - m_Evaluation.errorRate();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return acc;
			//fitness = obj.fit(acc, fs, fNum);
		}else {
			//fitness = 0;
			acc[0] = -1;
			acc[1] = -1;
			acc[2] = -1;
			return acc;
		}

	}

	
	public double[] getFitness(double[][] pos){
		//get the fitness of each individual in pos.
		double[] fitness = new double[pos.length];
		for(int i = 0; i < pos.length; i++) {
			fitness[i] = getFitness(pos[i]);
		}
		return fitness;

	}
}
