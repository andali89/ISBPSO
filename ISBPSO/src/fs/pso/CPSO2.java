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

import java.util.Random;

import fs.utils.Matcd;
import fs.utils.Wkc;
import weka.core.Instances;
import weka.core.Utils;

// compared with CPSO, this version initialize velocity as 0;
public class CPSO2 extends CPSO {

	public CPSO2(Instances data, Wkc classi) {
		super(data, classi);
	}
	public CPSO2(Instances data, Wkc classi, int popNum, int iterTime) {
		super(data, classi, popNum, iterTime);
	}
	
	@Override
	public void run()  {
		// run the sbpso to find the best solution
		System.out.println("CPSO2 used");
		this.resetPara();
		System.out.printf("The VThred is %f", this.vThred);
		long startTime = System.currentTimeMillis(); // start time
		double[][] pos = new double[popNum][fNum];
		double[][] vel = new double[popNum][fNum];
		newInitialization(pos, vel, popNum / 3, bitThred, rnd);
		double[] fitness = null;
		 
		try {
			fitness = this.getFitness(pos);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		double[] index = Matcd.maxVector(fitness);
		double[] gbest = pos[(int)index[0]].clone();
		double gbestFit = fitness[(int) index[0]];
		
       
        
		//copy the pos to best0 and pbest1
		double[][] pbest = new double[this.popNum][this.fNum];

		for(int i = 0; i < this.popNum; i++) {
			pbest[i] = pos[i].clone();

		}

		double[] pbestFit = fitness.clone();
		
		
		// getInfo
		
		iterNum[0] = Matcd.sumVector(pos[(int)index[0]]);
		this.fitZero = gbestFit;		
		iterMeanFit[0] = Matcd.meanVector(fitness);
		double[] posNums = getNum(pos);
		this.iterMeanNum[0] = Matcd.meanVector(posNums);
		double[] pbestNums = posNums.clone();
		this.iterMeanPbestFit = iterMeanFit.clone();
		this.iterMeanPbestNum = iterMeanNum.clone();
		double gbestNum = posNums[(int)index[0]];

		
		// start the iteration
		for(int i = 0; i < this.iterTime; i++) {
			//iteration stop until the number of iterations
			for(int j = 0; j < this.popNum; j++) {
				// for each individual in the population
				for(int k = 0; k < this.fNum; k++) {
					// for each element in the individual
					vel[j][k] =  W * vel[j][k] + C1 * rnd.nextDouble()*(pbest[j][k] - pos[j][k])
							+ C2 * rnd.nextDouble() * (gbest[k] - pos[j][k]);
					
					if(vel[j][k] > vThred) {
						vel[j][k] = vThred;
					}else if(vel[j][k] < -vThred){
						vel[j][k] = - vThred;
					}
					
					
					pos[j][k] = pos[j][k] + vel[j][k];
					
					// normarlize the position to [0 , 1]
					if(pos[j][k] > 1) {
						pos[j][k] = 1;
					}else if(pos[j][k] < 0) {
						pos[j][k] = 0;
					}
					
					
				}
				// get the fitness of individual j
				fitness[j] = this.getFitness(pos[j]);
				//System.out.printf("i th acc is %f", fitness[j]);
				posNums[j] = getNum(pos[j]);
				//update the gbest and pbest
				
				if(fitness[j] > pbestFit[j] ||
						(fitness[j] == pbestFit[j] && posNums[j] < pbestNums[j]) ) {
					pbest[j] = pos[j].clone();
					pbestFit[j] = fitness[j];
					pbestNums[j] = posNums[j];
				}

			}

			int cpbestIn = Utils.maxIndex(fitness);
			if(fitness[cpbestIn] > gbestFit || 
					(fitness[cpbestIn] == gbestFit && posNums[cpbestIn] < gbestNum)  ) {
				gbest = pos[cpbestIn].clone();
				gbestFit = fitness[cpbestIn];
				gbestNum = posNums[cpbestIn];

			}


			this.iterInfo[i] = gbestFit;
			this.iterNum[i + 1] = gbestNum;			
			
			this.iterMeanFit[i + 1] = Matcd.meanVector(fitness);
			this.iterMeanNum[i + 1] = Matcd.meanVector(posNums);
			this.iterMeanPbestFit[i + 1] = Matcd.meanVector(pbestFit);
			this.iterMeanPbestNum[i + 1] = Matcd.meanVector(pbestNums);
			
			
			System.out.printf("iteration time: %d, fitness value: %f\n", i + 1, iterInfo[i]);
		}
		this.gbest = gbest;
		this.gbestFit = gbestFit;
		//measure the runtime of the pso
		this.runtime = (System.currentTimeMillis() - startTime) / 1000.0;
	}
    
	/**
	 * Generate the swarm position and velocity, the small number of position is generated 
	 * as over 50 percent of features are selected, the large number of postion is generated 
	 * as 10 percent of features are selected. the velocity is generated in [-4, 4] randomly.
	 * 
	 * @param pos
	 * @param velocity
	 * @param smallnum
	 * @param thred
	 * @param rnd
	 */
	@Override
	public void newInitialization(double[][] pos, double[][] velocity, 
			int smallnum, double thred, Random rnd) {
		double scaleV = vThred;
		int fnum = pos[0].length;
		int popNum = pos.length;
		int largenum = popNum - smallnum;
		
		// for large part of swarm
		double[][] largeswarm = new double[largenum][fnum];
		
		for(int s = 0; s < largenum; s++) {
			for(int i = 0; i <fnum; i++) {
				if(rnd.nextDouble() < 0.1) {
					// select
					largeswarm[s][i] = thred +  rnd.nextDouble() * (1 - thred);
				}else {
					largeswarm[s][i] = 1 - (thred +  rnd.nextDouble() * (1 - thred));
				}
			}		
			
		}
		
		
		
		// for that in half to whole
		double[][] smallswarm = new double[smallnum][fnum];
		
		for(int s = 0; s < smallnum; s++) {
			
			int toselec = Matcd.randomInt(fnum / 2, fnum, rnd);
			// generate a set of weights, then sort to find first tosec number of features.
			double[] weights = new double[fnum];
			for(int i = 0; i < fnum; i++) {
				weights[i] = rnd.nextDouble();			
			}
			int[] index = Utils.sort(weights);
			for(int i = 0; i < fnum; i++) {
				if(i < toselec) {
					// to select the feature
					smallswarm[s][index[i]] = thred +  rnd.nextDouble() * (1 - thred);
				}else {
					smallswarm[s][index[i]] = 1 - (thred +  rnd.nextDouble() * (1 - thred));
				}				
			}			
			
		}
		
		// copy the generated large and small swarm to the pos
		for(int i = 0; i < popNum; i++) {
			if(i < largenum) {
				System.arraycopy(largeswarm[i], 0, pos[i], 0, fnum);
			}else {
				System.arraycopy(smallswarm[i - largenum], 0, pos[i], 0, fnum);
			}		
			
		}
		
		
		//initialize velocity as 0
		for(int i = 0; i < popNum; i++) {
			for(int j = 0; j < fnum; j++) {
				velocity[i][j] = 0.0;				
			}			
		}
		System.out.printf("The VThred is %f", scaleV);
		
	}

}
