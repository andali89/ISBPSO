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

package fs.pso;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instances;
import java.util.BitSet;
import java.util.Random;

import fs.utils.*;

public abstract class Absbpso implements PSO {
	
	/**
	 * store the fitness zero and iterNums of gbest
	 */
	protected double[] iterNum;
	protected double fitZero;
	
	/**
	 * store the mean fitness and num of swarms (with iteration 0)
	 */
	protected double[] iterMeanFit;
	protected double[] iterMeanNum;
	
	/**
	 * stor the fitness and number of pbest (with iteration 0)
	 */
	protected double[] iterMeanPbestFit;
	protected double[] iterMeanPbestNum;
	///////
	protected Instances data;
	public Objfuc obj;
	public Wkc classi;
	protected int insNum;
	protected int fNum; //full feature number
	protected int popNum;
	protected int iterTime;
	protected int stickNum;
	protected double[] impg; // store the im, ip and ig.
	public double[] iterInfo; // iterNum and fitness.
	public double[] gbest;
	public double gbestFit;
	public double runtime;
	protected int innerfold = 5;	
	ASEvaluation ASEval;
	protected double[] objWeight = {0.9, 0.1};
	protected Random rnd;
	public Absbpso(Instances datain, Wkc classi) {
		// TODO Auto-generated constructor stub
		this.data = new Instances(datain);
		try {
			//this.data = Basicfunc.norm(data);
			//this.data = Basicfunc.standardizer(data);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
			this.data = new Instances(datain);
		}//this.data = new Instances(data);
		fNum = this.data.numAttributes() -1;
	
		try {
			this.classi = classi;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		rnd = new Random(1);
		
	}
	public void setEvaluator(ASEvaluation ASEval) {
		this.ASEval = ASEval;
	}
	public void setSeed(long seed) {
		this.rnd = new Random(seed);
	}

	
	public void setPara(int popNum, int iterTime, int stickNum, double[] impg, int objtype) {
		// set the parameters of the sbpso
		this.popNum = popNum;
		this.impg = impg;
		this.iterTime = iterTime;
		this.stickNum = stickNum;
		this.obj = new Objfuc(objtype);
		obj.setWeight(this.objWeight[0], this.objWeight[1]);
		//((ModiWrapperSubsetEval)subsetEval).obj = obj;
	}
	
	public void setObjWeight(double[] W) {
		objWeight = W;
		this.obj.setWeight(W[0], W[1]);
	}
	public abstract void run();
	
	protected double[] getNum(double[][] pos) {
		// get the number of selected features in positions
		int num = pos.length;
		double[] Nums = new double[num]; 
		for(int i = 0; i < num; i++) {
			Nums[i] = getNum(pos[i]);
			
		}
		return Nums;
	}
	
	public double getNum(double[] pos) {
		int Num = 0;
		for(int i = 0; i < pos.length; i++) {
			Num += (pos[i] >= 0.5? 1 : 0);
		}
		return Num;
		
	}
	
	public double getFitness(double[] pos) {
		double acc = 0;
		BitSet bitset = new BitSet(fNum + 1);
		for(int i = 0; i < fNum; i++) {
			if(pos[i] >= 0.5) {
				bitset.set(i);
			}else {
				bitset.clear(i);
			}
		}
		//bitset.clear(fNum);	
	
		try {
			acc = ((SubsetEvaluator) ASEval).evaluateSubset(bitset);
			
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		return acc;
	}
//	public double getFitness(double[] acc,Objfuc obj)  {
//		// get the fitness of the obj, fs denote the number			
//		return  obj.fit(acc, (int) acc[0], fNum);
//		
//	}
//	protected double[] getAcc(double[] pos) throws Exception {		
//		
//		double[] acc = new double[3];
//		Random Rnd = new Random(1);
//		Instances datain = Basicfunc.select(this.data, pos);
//		int fs = datain.numAttributes() -1;		
//		m_Evaluation = new Evaluation(datain);
//		if(fs > 0) {
//			classi.setData(datain, innerfold);			
//			acc[0] = fs;
//			acc[1] =0.5;
//			try {
//				m_Evaluation.crossValidateModel((Classifier)this.classi.classifier, datain, innerfold,
//						Rnd);
//				//acc = classi.cvRun();
//				acc[2] = 1 - m_Evaluation.errorRate();
//			} catch (Exception e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//			return acc;
//			//fitness = obj.fit(acc, fs, fNum);
//		}else {
//			//fitness = 0;
//			acc[0] = -1;
//			acc[1] = -1;
//			acc[2] = -1;
//			return acc;
//		}
//		
//	}
	
	public void setInnerfold(int innerfold) {
		this.innerfold = innerfold;
	}
	public double[] getFitness(double[][] pos) throws Exception {
		//get the fitness of each individual in pos.
		double[] fitness = new double[pos.length];
		for(int i = 0; i < pos.length; i++) {
			fitness[i] = getFitness(pos[i]);
		}
		return fitness;
		
	}
	
	
	public double[] gbest() {
		return this.gbest;
	}
	public double gbestFit() {
		return this.gbestFit;
	}
	public double[] iterInfo() {
		return this.iterInfo;
	}
	public double runtime() {
		return this.runtime;
	}
	public double[] iterNum() {
		return this.iterNum;
	}
	public double fitZero() {
		return this.fitZero;
	}
	
	public double[] iterMeanFit() {
		return this.iterMeanFit;
	}
	
	public double[] iterMeanNum() {
		return this.iterMeanNum;
	}
	
	public double[] iterMeanPbestFit() {
		return this.iterMeanPbestFit;
	}
	public double[] iterMeanPbestNum() {
		return this.iterMeanPbestNum;
	}
	

}
