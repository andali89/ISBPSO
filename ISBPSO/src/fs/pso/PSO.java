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

public interface PSO {
	public void run();
	public double[] gbest();
	public double gbestFit();
	public double runtime();
	public double[] iterInfo();
	public void setSeed(long rnd);
	public void setEvaluator(ASEvaluation ASEval) ;
	public double[] iterNum();
	public double fitZero() ;
	public double[] iterMeanFit();	
	public double[] iterMeanNum() ;	
	public double[] iterMeanPbestFit();
	public double[] iterMeanPbestNum(); 
	public double getNum(double[] pos);
}
