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

public class Matcd {
	
	public static double[][] iniPop(int popNum, int fLength){
		// initilized as 0 and 1 like [0,1,0,1,1]
		Random rnd = new Random(Math.round(Math.random()*10000));
		return iniPop(popNum, fLength, rnd);
	}
	
	public static double[][] iniPop(int popNum, int fLength, Random rnd){
		double[][] pop = new double[popNum][fLength]; 
		for(int i = 0; i < popNum; i++ ) {
			for(int j = 0; j < fLength; j++) {
				if(rnd.nextDouble() < 0.5) {
					pop[i][j] = 0;					
				}else {
					pop[i][j] = 1;
				}
			}
		}
		return pop;
	}
	
	public static double[][] iniPopReal(int popNum, int fLength){
		// initilized as 0 and 1 like [0.1,0.9,0,1,1]
		Random rnd = new Random(Math.round(Math.random()*10000));
		
		return iniPopReal(popNum, fLength, rnd);
	}
	public static double[][] iniPopReal(int popNum, int fLength, Random rnd){
		// initilized as 0 and 1 like [0.1,0.9,0,1,1]
		double[][] pop = new double[popNum][fLength]; 
		for(int i = 0; i < popNum; i++ ) {
			for(int j = 0; j < fLength; j++) {
				
					pop[i][j] = rnd.nextDouble();					
				
			}
		}
		return pop;
	}
	
	public static double[][] iniPop(int popNum, int fLength, double[] score) throws Exception{
		Random rnd = new Random(Math.round(Math.random()*100));
		return iniPop(popNum,fLength, score, rnd);
	}
	
	public static double[][] iniPop(int popNum, int fLength, double[] score, Random rnd) throws Exception{
		// initilized as 0 and 1 like [0,1,0,1,1]
		
		if(fLength != score.length) {
			throw new Exception("score length not equal to feature number");			
		}
		double[][] pop = new double[popNum][fLength]; 
		for(int i = 0; i < popNum; i++ ) {
			for(int j = 0; j < fLength; j++) {
				if(rnd.nextDouble() < score[j]) {
					pop[i][j] = 1;					
				}else {
					pop[i][j] = 0;
				}
			}
		}
		return pop;
	}
	
	public static double[][] sameNums(int rows, int cols, double num){
		//set the array with the same num
		double[][] setNums = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				setNums[i][j] = num;
			}
		}
		return setNums;
	}
	
	public static double[] sameNums(int rowLength, double num) {
		//set the array with the same num
		double[] setNums = new double[rowLength];
		for(int i = 0; i < rowLength; i++) {
			setNums[i] = num;
		}
		return setNums;
	}
	
	public static double[][] maxMatrix(double[][] matrix, int dim) {
		//dim == 0 get the max number of each row
		//dim == 1 get the max number of each column
		//maxNum have [2][x], [0][:] store the index of max,[1][:] store the number
		int rowNum = matrix.length;
		int colNum = matrix[0].length;	
		double[][] maxNum = null;
		if(dim == 0) {
			maxNum = new double[2][rowNum];
			for(int i = 0; i < rowNum; i++) {
				maxNum[0][i] = 0;
				maxNum[1][i] = matrix[i][0];
				for(int j = 1; j < colNum; j++) {
					if(maxNum[1][i] < matrix[i][j]) {
						maxNum[1][i] = matrix[i][j];
						maxNum[0][i] = j;
					}
				}
			}			
		}
		// get the max number of each column
		if(dim == 1) {
			maxNum = new double[2][colNum];
			for(int j = 0; j < colNum; j++) {
				maxNum[0][j] = 0;
				maxNum[1][j] = matrix[0][j];
				for(int i = 1; i < rowNum; i++) {
					// get the max number in row i
					if(maxNum[1][j] < matrix[i][j]) {
						maxNum[0][j] = i;
						maxNum[1][j] = matrix[i][j];
					}
					 
				}
			}					
		}
		return maxNum;
		
	}
	
	public static double[] maxVector(double[] vector) {
		// find the index and number in a vector
		double[] maxNum = new double[2];
		maxNum[0] = 0;
		maxNum[1] = vector[0];
		for(int i = 1; i < vector.length; i++) {
			if(maxNum[1] < vector[i]) {
				maxNum[0] = i;
				maxNum[1] = vector[i];
			}
		}
		return maxNum;
	}
	
	public static double sumVector(double[] vector) {
		// sum of elements of the vector
		double sum = 0;
		for(int i = 0; i < vector.length; i++) {
			sum += vector[i];
		}
		return sum;
	}
	
	public static double meanVector(double[] vector) {
		// mean value of the vector
		return sumVector(vector)/vector.length;
	}
	
	/**
	 * dim == 0 get the mean number of each row
	 * dim == 1 get the mean number of each column
	 * @param matrix
	 * @param dim
	 * @return
	 * @throws Exception
	 */
	public static double[] meanMatrix(double[][] matrix, int dim) throws Exception {
		
		int rowNum = matrix.length;
		int colNum = matrix[0].length;
		double[] mn;
		if(dim == 0) {
			mn = new double[rowNum];
			for(int i = 0; i < rowNum; i++) {
				mn[i] = meanVector(matrix[i]);
			}			
		}else if(dim == 1) {
			mn = new double[colNum];
			for(int j = 0; j < colNum; j++) {
				mn[j] = 0;
				for(int i = 0; i < rowNum; i++) {
					mn[j] +=matrix[i][j];
				}
				mn[j] = mn[j] / rowNum;
			}
		}else {			
			throw new java.lang.Exception("dim must be 0 or 1");
		}
		return mn;
	}
	
	public static double stdVector(double[] vector, int type) {
		//type = 0 then n type = 1 then n-1
		double mn = 0; //mean value
		double sqm = 0; //mean value of x^2
		for(int i = 0; i < vector.length; i++) {
			mn += vector[i];
			sqm += Math.pow(vector[i], 2);
		}
		double re = Math.pow(sqm / vector.length - Math.pow(mn/ vector.length, 2), 0.5);
		if(type == 1) {
			// given the sample calculate the all
			re = re * Math.sqrt(vector.length / (vector.length -1.0));
		}
		return re;
	}
	
	public static double[] stdMatrix(double[][] matrix, int type, int dim) throws Exception {
		// dim = 0 calculate the std for each row
		// dim = 1 calculate the std for each column
		// type = 0 then n type = 1 then n-1
		int row = matrix.length;
		int col = matrix[0].length;
		double[] std;
		if(dim == 0) {
			std = new double[row];			
			for(int i = 0; i < row; i++) {
				std[i] = stdVector(matrix[i], type);
			}
		}else if (dim == 1) {
			std = new double[col];
			double mn;
			double sqm;
			for(int j = 0; j < col; j++) {
				mn = 0;
				sqm = 0;
				for(int i = 0; i < row; i++) {
					mn += matrix[i][j];
					sqm += Math.pow(matrix[i][j], 2);
				}
				std[j] = Math.pow(sqm / row - Math.pow(mn / row, 2), 0.5);
				if(type == 1) {
					std[j] = std[j] * Math.sqrt(row / (row -1.0));
				}
			}
			
		}else {
			throw new java.lang.Exception("dim must be 0 or 1");
		}
		return std;
	}
	
	public static double[] mergeVector(double[] vectorA, double[] vectorB) {
		int length = vectorA.length + vectorB.length;
		double[] vectorO = new double[length];
		for(int i = 0; i < vectorA.length; i++) {
			vectorO[i] = vectorA[i];
		}
		
		int j = vectorA.length;
		for(int i = 0; i < vectorB.length; i++) {
			vectorO[j] = vectorB[i];
			j++;
		}
		return vectorO;
	}
	/**
	 *  return a random int between a(inclusive) and b(inclusive)
	 */  
	public static int randomInt(int a, int b) {
		Random rnd = new Random(Math.round(Math.random()*10000));
		return randomInt(a, b, rnd);
	}
	public static int randomInt(int a, int b, Random rnd) {
		int reNum = a;
		if(a != b) {
			double value = rnd.nextDouble();
			reNum = (int) Math.floor((a + (b + 1 - a) * value));
			
		}
		return reNum;
	}
	
	/**
	 * 
	 * @param list
	 * @param startIndex
	 * @param num if num == -1, then selects startIndex to the end
	 * @return
	 */
	public static int[] slice(int[] list, int startIndex, int num) {
		int [] sliced = new int[num];
		if (num == -1 || (startIndex + num) > list.length) {
			num = list.length - startIndex;
		}
		for (int i = 0; i < num; i++) {
			sliced[i] = list[startIndex + i];
		}
		
		return sliced;	
		
	}
}
