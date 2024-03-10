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

import java.util.List;
import java.util.Random;

import fs.utils.Matcd;

public class GAOp {
	
    private double crate;
    private double mrate;
    private int row;
    private int column;
    private Random rnd;
	public GAOp(Random rnd) {
		// TODO Auto-generated constructor stub
		this.rnd = rnd;
		resetOptions();
	}
	public void resetOptions(){
		crate = 0.9;
		mrate = 0.01;
	}
	
	public void operate(double[][] pos,double[][]posM, double[] mask, double[] fitness, List<Integer> retainFeatures) {
		// element of masked position is 0
		row = pos.length;
		column = pos[0].length;
		mrate = 1 / Matcd.sumVector(mask);
		cross(pos, posM, fitness, retainFeatures);
		mutation(posM, mask);
	}
	private void mutation(double[][] posM, double[] mask) {
		// TODO Auto-generated method stub
		for(int i = 0; i < row; i++) {
			for(int j = 0; j < column; j++) {
				if(rnd.nextDouble() < mrate && mask[j] == 1) {
					posM[i][j] = 1 - posM[i][j];
				}
			}
		}
		
	}
	private void cross(double[][] pos, double[][]posM, double[] fitness, List<Integer> retainFeatures) {
		int index1;
		int index2;
		int changepoint;
		int retainNums = retainFeatures.size();
		if(retainNums <=1) {
			return;
		}
		for(int i = 0; i < row; i += 2) {
			index1 = binaryTour(fitness);
			do {
				index2 = binaryTour(fitness);
			}while(index1 == index2);
			//changepoint = Matcd.randomInt(1, column - 1, rnd);
			changepoint = retainFeatures.get(Matcd.randomInt(1, retainNums - 1, rnd));
			
			if(rnd.nextDouble() < crate) {				
			    System.arraycopy(pos[index1], 0, posM[i], 0, changepoint);
			    System.arraycopy(pos[index2], 0, posM[i + 1], 0, changepoint);
			    System.arraycopy(pos[index2], changepoint, posM[i], changepoint, column - changepoint);
			    System.arraycopy(pos[index1], changepoint, posM[i +1], changepoint, column - changepoint);
			}else {
				 System.arraycopy(pos[index1], 0, posM[i], 0, column);
				 System.arraycopy(pos[index2], 0, posM[i + 1], 0, column);
			}
			
		}
	}
	
	
	
	/**
	 *
	 * @param vector store the fitness value
	 * @return the index with max vector value
	 */
	protected int binaryTour(double[] vector) {
		int num = (int) Math.floor(rnd.nextDouble() * vector.length);
		int num2;
		while(true) {
			num2 = (int) Math.floor(rnd.nextDouble() * vector.length);
			if(num2 != num){
				break;
			}
		}
		if(vector[num] > vector[num2]) {
			return num;
		}else {
			return num2;
		}
	}

}
