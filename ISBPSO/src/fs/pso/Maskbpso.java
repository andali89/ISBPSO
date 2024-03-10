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

import java.util.ArrayList;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PKIDiscretize; 
import java.util.List;

import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.core.Instances;
import weka.core.Utils;
import fs.utils.*;
public class Maskbpso extends Absbpso {

	protected int[] maskset;
	protected double elimipercent = 0.1; // the percent of eliminated, 0 in the paper
	protected List<Integer> retainFeatures = new ArrayList<Integer>();
	protected int currentFnum;
	protected String iniMethod = "random"; //relieff and ig
	//private double[] scorecheck;
	public double[] sigPara = new double[] { 0.1, 0.6, 6};
	public boolean inihalf = false; // revised 2020/5/6
	boolean GAop = false;
	boolean Maskop = true;
	public int NonCount = 5; //if NonCount >=5 perform GA
	public double dividePop = 3.0;
	//constructor
	public Maskbpso(Instances data, Wkc classi) {
		// TODO Auto-generated constructor stub
		super(data, classi);
	}
	
	public Maskbpso(Instances data, Wkc classi, int popNum, int iterTime, int stickNum, double[] impg, int objtype,int[] maskset) {
		super(data, classi);
		this.setPara(popNum, iterTime, stickNum, impg, objtype, maskset);
	}
	
	// overload the setPara function
	public void setPara(int popNum, int iterTime, int stickNum, double[] impg, int objtype, int[] maskset) {
		this.setPara(popNum, iterTime, stickNum, impg, objtype);
		this.maskset = maskset;
	}
	public void setElimPercent(double pc) {
		this.elimipercent = pc;
	}
	public void setiniMethod(String method) {
		this.iniMethod = method;
	}
	public void setGAop(boolean op, int nonCount) {
		this.GAop = op;
		this.NonCount = nonCount;
	}
	public void setMaskop(boolean op) {
		this.Maskop = op;
	}
	public void reSetCurrent() {
		this.currentFnum = this.fNum;
		retainFeatures.clear();
		for(int i = 0; i < this.fNum; i++) {
			this.retainFeatures.add(i);		
		}
	}
	
	@Override
	public void run() {
		//runtime
		System.out.println("HYMASKPSO");
		long startTime = System.currentTimeMillis();
		this.iterInfo = new double[this.iterTime];
		//double[][] pos = Matcd.iniPop(this.popNum, this.fNum);
		double[][] pos = null;
		reSetCurrent();
		try {
			pos = popIni();
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		double[][] stickness = Matcd.sameNums(this.popNum, this.fNum, 1);
		double stickStep = 1.0/this.stickNum;
		double[] fitness = null;
	
		try {
			fitness = this.getFitness(pos);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		int index = Utils.maxIndex(fitness);
		//double[] index = Matcd.maxVector(fitness);
		double[] gbest = pos[index].clone();
		double gbestFit = fitness[index];
		
		
		
		// getInfor
		this.iterNum = new double[iterTime + 1];
		iterNum[0] = Matcd.sumVector(pos[index]);
		this.fitZero = gbestFit;
		this.iterMeanFit = new double[iterTime + 1];
		iterMeanFit[0] = Matcd.meanVector(fitness);
		double[] posNums = getNum(pos);
		this.iterMeanNum = new double[iterTime +1];
		this.iterMeanNum[0] = Matcd.meanVector(posNums);
		double[] pbestNums = posNums.clone();
		this.iterMeanPbestFit = iterMeanFit.clone();
		this.iterMeanPbestNum = iterMeanNum.clone();
		double gbestNum = posNums[index];
		
		
		
		
		System.out.printf("iteration time: %d, fitness value: %f\n", 0, gbestFit);
		double[] pbestScore = getpbestScore(pos);
		double gbestScore = pbestScore[index];
		//copy hte pos to best0 and pbest1
		double[][] pbest = new double [this.popNum][this.fNum];
		for(int i = 0; i < this.popNum; i++) {
			pbest[i] = pos[i].clone();

		}
		double[] pbestFit = fitness.clone();
		//initialize the mask
		double[] mask = Matcd.sameNums(this.fNum, 1);
		int maskIndex = 0;		
		double pd = 0;
		boolean suc = false; // If the iteration suc	
		int nonUpcount = 0;
		int upMuRe = 0;
		//int cpbestIn = 0;
		// start the iteration
		for(int i = 0; i < this.iterTime; i++) {
			//iteration stop until the number of iterations		
			if(this.Maskop && maskIndex < maskset.length && i == Math.abs(this.maskset[maskIndex])) {
				//update the mask and pos
				//return new mask and update the pos
		        //the elements of masked position set to 0
		        // if the itertime <0 then unlock the mask at abs(itertime)
		        //0 denote the element is masked
				updateMaskPos(mask, gbest, pbest, pos, stickness, maskset[maskIndex]);
				maskIndex++;				
			}
			// a GA process conducted if not updated for NonCuont iterations.
			if(this.GAop && nonUpcount >= this.NonCount) {
//				System.out.print("gA process");
				upMuRe = updateGA(pbest, pbestFit, gbestFit,  mask);
				//update successful
				if(upMuRe != -1) {
					gbestFit = pbestFit[upMuRe];
					gbest = pbest[upMuRe];
					
				}
				iterInfo[i] = gbestFit;				
				
				iterNum[i + 1] = getNum(gbest);
				iterMeanFit[i + 1] = iterMeanFit[i];
				iterMeanNum[i + 1] = iterMeanNum[i];
				iterMeanPbestFit[i + 1] = Matcd.meanVector(pbestFit);
				pbestNums = getNum(pbest);
				iterMeanPbestNum[i + 1] = Matcd.meanVector(pbestNums);	
				
				
				nonUpcount = 0;
				continue;
			}
			suc = false;
			for(int j = 0; j < this.popNum; j++) {
				// for each individual in the population
				
				for(int k = 0; k < this.fNum; k++) {
					// for each element in the individual
					if(mask[k] ==1) {
						double p = rnd.nextDouble();
						pd = impg[0] * (1 - stickness[j][k]) + impg[1] * Math.abs(pbest[j][k] - pos[j][k]) + impg[2] * Math.abs(gbest[k] - pos[j][k]);
						//pd =  impg[1] * Math.abs(pbest[j][k] - pos[j][k]) + impg[2] * Math.abs(gbest[k] - pos[j][k]);
						if(p < pd) {
							//mutation
							
							pos[j][k] = 1 - pos[j][k];
							stickness[j][k] = 1;
						}else if(stickness[j][k] >= stickStep) {
							// update the stickness, stickness always >= 0
							stickness[j][k] -= stickStep;
						}
//						p = rnd.nextDouble();
//						if(p < 0.01) {
//							pos[j][k] = 1 - pos[j][k];
//						}
						
//						if (p <= impg[0]) {
//							// nothing to do!
//						} else if (p <= (impg[0] + impg[1])) {
//							pos[j][k] = gbest[k];
//						} else {
//							pos[j][k] = pbest[j][k];
//						} 
//						p = rnd.nextDouble();
//						if(p < 0.01) {
//							pos[j][k] = 1 - pos[j][k];
//						}
						
					}
					
				}
				// get the fitness of individual j
				fitness[j] = this.getFitness(pos[j]);
				posNums[j] = getNum(pos[j]);
				//update the gbest and pbest
//				double[] sucPbFit = new double[3];
//				sucPbFit = updatePbGb (j, pos, fitness, pbest, gbest, pbestFit, 
//						gbestFit, pbestScore, gbestScore);
//				gbestFit = sucPbFit[1];
//				gbestScore = sucPbFit[2];
				
				//steady version
//				if(fitness[j] > pbestFit[j]) {
//					pbest[j] = pos[j].clone();
//					pbestFit[j] = fitness[j];
//					
//				}else if(fitness[j] == pbestFit[j]) {
//					double score = this.getgbestScore(pos[j]);
//					if(score > pbestScore[j]) {
//						pbest[j] = pos[j].clone();
//						pbestFit[j] = fitness[j];
//						pbestScore[j] = score;
//					}	
//				}
				//steady version end
				
				if(fitness[j] == gbestFit) {
					double score = this.getgbestScore(pos[j]);
					if(score > gbestScore) {
						pbest[j] = pos[j].clone();
						pbestFit[j] = fitness[j];
						gbest = pos[j].clone();
						gbestFit = fitness[j];
						gbestScore = score;
						suc = true;
						//
						pbestNums[j] = posNums[j];
						gbestNum = posNums[j];
						
					}					
				}else if(fitness[j] == pbestFit[j]) {
					double score = this.getgbestScore(pos[j]);
					if(score > pbestScore[j]) {
						pbest[j] = pos[j].clone();
						pbestFit[j] = fitness[j];
						pbestScore[j] = score;
						//
						pbestNums[j] = posNums[j];
					}	
				}else {
					if(fitness[j] > gbestFit) {
						pbest[j] = pos[j].clone();
						pbestFit[j] = fitness[j];
						gbest = pos[j].clone();
						gbestFit = fitness[j];
						suc = true;
						//
						pbestNums[j] = posNums[j];
						gbestNum = posNums[j];
						
					}else if(fitness[j] > pbestFit[j]) {
						pbest[j] = pos[j].clone();
						pbestFit[j] = fitness[j];
						//
						pbestNums[j] = posNums[j];
					}
				}	
				
				
			}
			// steady version
//			cpbestIn = Utils.maxIndex(fitness);
//			if(fitness[cpbestIn] > gbestFit || 
//					(fitness[cpbestIn] == gbestFit && pbestScore[cpbestIn] > gbestScore)) {
//				gbest = pos[cpbestIn].clone();
//				gbestFit = fitness[cpbestIn];
//				gbestScore = pbestScore[cpbestIn];
//				suc = true;
//			}
			// steady version end
			
			System.out.println("current musk number is " + String.valueOf(Matcd.sumVector(mask)));
			
			nonUpcount = suc ?  0 : (nonUpcount +1); 
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
		this.runtime = (System.currentTimeMillis() - startTime)/1000.0; //measure the iteration time of pso
	}
	
	protected double[] getpbestScore(double[][] pos) {
		double[] score = new double[pos.length];
		for(int i = 0; i < pos.length; i++) {
			score[i] = getgbestScore(pos[i]);
		}
		return score;
	}
	
	public double getgbestScore(double[] gbest) {
		 
		double sumScore = fNum - Matcd.sumVector(gbest);
//		if(this.iniMethod.equals("random")) {
//			return fNum - Matcd.sumVector(gbest);
//		}
//		for(int i = 0; i < this.fNum; i++) {
//			if(gbest[i] == 1.0) {
//				
//				sumScore += this.scorecheck[i] / 100 - 100;
//				
//			}
//		}
		return sumScore;
	}

	@SuppressWarnings("unused")
	private double[] updatePbGb(int j, double[][] pos, double[] fitness,double[][] pbest, double[] gbest,
			double[] pbestFit, double gbestFit, double[] pbestScore, double gbestScore) {
		// sucif, gbestFit, gbestScore	
		double posscore = this.getgbestScore(pos[j]);
		double[] re = new double[3];
		if(fitness[j] > gbestFit ) {			
			re[0] = 2;
		}else if(fitness[j] >= pbestFit[j]) {
			re[0] = 1;			
		}else {
			re[0] = 0;
		}
		if(re[0] == 2) {
			pbest[j] = pos[j].clone();
			pbestFit[j] = fitness[j];
			gbest = pos[j].clone();
			gbestFit = fitness[j];
			pbestScore[j] = posscore;
			gbestScore = posscore;
		}else if(re[0] == 1){
			pbest[j] = pos[j].clone();
			pbestFit[j] = fitness[j];
			pbestScore[j] = posscore;
		}
		re[1] = gbestFit;
		re[2] = gbestScore;
		return re;
	}

	protected void updateMaskPos(double[] mask,double[] gbest,double[][] pbest,double[][] pos, double[][] stickness, int maskNum) {
		if(maskNum >= 0) {
			double selNum;
			int count = 0;
			retainFeatures.clear();			
			for(int i = 0; i < this.fNum; i++) {
				selNum = 0;
				for(int j = 0; j < pbest.length; j++) {
					selNum += pbest[j][i];
				}				
				if(gbest[i] != 1 && selNum <= (double)this.popNum * this.elimipercent) {
				//if the number that the feature is selected less than elimipercen of the whole number of pop
			    // and this feature is not selected by gbest then the one is masked, the pos is masked
					count ++; // denote the reduced feature			
					
					mask[i] = 0;
					for(int k = 0; k < pbest.length; k++) {
						pos[k][i] = 0;
						stickness[k][i] = 0;
					}
				}else{
					retainFeatures.add(i);
				}
			}
			currentFnum = this.fNum - count;
			System.out.println(count);
		}else {
			System.out.println(" all the mask is eliminated");
			// all the mask is eliminated
			for(int i = 0; i < this.fNum; i++) {
				mask[i] = 1;
				retainFeatures.add(i);
			}
		}
	}
	// for test
	protected void CountMaskPos(double[] mask,double[] gbest,double[][] pbest,double[][] pos, double[][] stickness, int maskNum) {

		double selNum;
		int count = 0;
				
		for(int i = 0; i < this.fNum; i++) {
			selNum = 0;
			for(int j = 0; j < pbest.length; j++) {
				selNum += pbest[j][i];
			}		
			//System.out.println("selNUm" + String.valueOf(selNum));
			if(gbest[i] != 1 && selNum <= (double)this.popNum * this.elimipercent) {
				//if the number that the feature is selected less than 1/10 of the whole number of pop
				// and this feature is not selected by gbest then the one is masked, the pos is masked
				count ++; // denote the reduced feature			


			}
		}			
		System.out.println(count);


	}
	
	
	
	protected double[][] popIni() throws Exception{
		double[][] pos;
		if(iniMethod.equals("random")) {
			//pos = new double[popNum][fNum];
			//initPopulation(pos);
			pos = Matcd.iniPop(popNum, fNum, rnd);
		}else {
			System.out.println("new initialization method used!");
			double[] score = getWeight(this.iniMethod);

//			int[] inde = weka.core.Utils.sort(score);
////			for(int i=0; i<fNum;i++) {
////				System.out.println(score[inde[fNum - i - 1]]);
////			}
			if (inihalf) {
				System.out.printf("divdePop %f \r\n", dividePop);
				int halfpopNum = (int) (popNum / dividePop);
				double[][] pos1 = Matcd.iniPop(halfpopNum, this.fNum , score, rnd);
				//double[][] pos1 = Matcd.iniPop(halfpopNum, this.fNum , score);
				//double[][] pos2 = new double[this.popNum - halfpopNum][fNum];
				//initPopulation (pos2);
				System.out.println("aaa");
				double[][] pos2 = Matcd.iniPop(this.popNum - halfpopNum, fNum, rnd);
				pos = new double[popNum][fNum];
				for(int i = 0; i < halfpopNum; i++) {
					pos[i] = pos1[i];
				}
				for(int i = halfpopNum;  i < popNum ; i++) {
					pos[i] = pos2[i - halfpopNum];
				}
				System.out.print("abc");
			}else {
				pos = Matcd.iniPop(popNum, this.fNum , score);
				//pos = new double[popNum][fNum];
				//initPopulation(pos);
			}
			
		}

		return pos;
	}
	
	@SuppressWarnings("unused")
	private void initPopulation (double pos[][]) throws Exception {
		int i,j,bit;
		int num_bits;
		boolean ok;
		int start = 0;	
		int m_numAttribs = this.fNum + 1;
        //Random rnd = new Random(3);
		for (i=start;i<pos.length;i++) {			
            pos[i] = Matcd.sameNums(fNum, 0);
			num_bits = rnd.nextInt();
			num_bits = num_bits % (this.fNum + 1)-1;
			if (num_bits < 0) {
				num_bits *= -1;
			}
			if (num_bits == 0) {
				num_bits = 1;
			}

			for (j=0;j<num_bits;j++) {
				ok = false;
				do {
					bit = rnd.nextInt();
					if (bit < 0) {
						bit *= -1;
					}
					bit = bit % m_numAttribs;
					
					if (bit != fNum) {
							ok = true;
					}
					
					
				} while (!ok);

				if (bit > m_numAttribs) {
					throw new Exception("Problem in population init");
				}
				pos[i][bit] = 1;
				
			} // for num_bits
//			for(j = 0; j < 5; j++) {
//				pos[i][j] = 1;
//			}
			
		
		} // for popSize
	} // initPopulation
	
	
	
	
	
	

	private double[] getWeight(String iniMethod) throws Exception {
		double[] score = new double[this.fNum];
		AttributeEvaluator eval = null;
		if(iniMethod.toLowerCase().equals("hybrid")) {
			double[] score1 = getWeight("relieff");
			double[] score2 = getWeight("ig"); // note that the last one is store as the store check
			if (score2[0] == 0.49909123) {
				System.out.println("ok!");
				for(int i = 0; i < fNum; i++) {
					score[i] = 0.5 * score1[i];    				
				}	
			}else {
				for(int i = 0; i < fNum; i++) {
					score[i] = 0.5 * (score1[i] + score2[i]);
					//score[i] = (score1[i] > score2[i])? score1[i] : score2[i];
				}	
			}

		}else {
			if(iniMethod.toLowerCase().equals("acc")) {
				Objfuc obj = new Objfuc(0);
				obj.setWeight(1, 0);
				double[] pos = Matcd.sameNums(fNum, 0);
				pos[0] = 1;
				//score[0] = getFitness(pos, obj);
				for(int i = 1; i < fNum; i++) {
					pos[i - 1] = 0;
					pos[i] = 1;
					//score[i] = getFitness(pos, obj);
				}


			}else {
				switch(iniMethod.toLowerCase()) {
				case "relieff":
					eval = new ReliefFAttributeEval();
					((ReliefFAttributeEval) eval).buildEvaluator(this.data);
					System.out.println("relieff");
					break;
				case "su":
					eval = new SymmetricalUncertAttributeEval();
					((SymmetricalUncertAttributeEval) eval).buildEvaluator(this.data);
					break;
				case "ig":
					eval = new InfoGainAttributeEval();
					((InfoGainAttributeEval) eval).buildEvaluator(this.data);
					System.out.println("ig");
					break;
				}
				for(int i = 0; i < fNum; i++) {
					score[i] = eval.evaluateAttribute(i);

				}

			}



			//this.scorecheck = score;
			double tempNormScore;
			double maxScore = score[weka.core.Utils.maxIndex(score)];
			double minScore = score[weka.core.Utils.minIndex(score)];
			double scale = 2 * sigPara[2] / (maxScore - minScore);
			for(int i = 0; i < fNum; i++) {
				tempNormScore = -sigPara[2] + scale * (score[i] - minScore);
				score[i] = sigPara[0] + (sigPara[1] - sigPara[0]) / (1.0 + Math.exp(- tempNormScore));

				//System.out.println(String.valueOf(score[i]));
			}
		}

		if(Double.isNaN(score[0])) {
//			for(int i = 0; i < fNum; i++) {
//				score[i] = 0.49909123;
//			}
			getweightForIG(iniMethod, score);
			System.out.println("PKI INITIALIZATION METHOD USED");
		}

		return score;

	}
	
	
	
	
	 
	/**
	 * 
	 * @param pbest
	 * @param pbestFit
	 * @param gbest
	 * @param gbestFit
	 * @param mask
	 * @return
	 * return the index of gbest0 if index = -1 means the gbest do not change
	 */
	protected int updateGA(double[][] pbest, double[] pbestFit,
			 double gbestFit, double[] mask)  {
		int gbMindex = -1; //-1 denote not change	       
		double[][]posM = new double[pbest.length][pbest[0].length];
		GAOp gaop = new GAOp(rnd);
		gaop.operate(pbest, posM, mask, pbestFit, retainFeatures);
		int popLen = pbest.length;

		double[] posMfits =null;
		try {
			posMfits = getFitness(posM);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} //1 by pNum
		// put the original fitness and mutated fitness in wFit0 and wFit1
		double[] wFit = new double[2 * popLen];
		
		System.arraycopy(pbestFit, 0, wFit, 0, popLen);
		System.arraycopy(posMfits, 0, wFit, popLen, popLen);			

		System.out.println("fit");
		gbMindex = upInnerPbest(wFit, pbest, pbestFit, posM, popLen, gbestFit);


		return gbMindex;

	}
	//inner function of updateMutation, to update pbest0 or pbest1 according to the Fitness
	private int upInnerPbest(double[] wFit0, double[][] pbest0, double[] pbestFit0, double[][] posM, int popLen, double gbestFit0) {
		int bestIndex = 0;
		int[] index0 = weka.core.Utils.sort(wFit0);	//wFit0 2*popLen by 1
		if(wFit0[index0[2 * popLen -1]] > gbestFit0) {
			System.out.println("dfsafdsfdsfsadfsfd");
		}
		List<Integer> keepFit0 = new ArrayList<Integer>();
		List<Integer> nokeepFit0 = new ArrayList<Integer>();
		// add the old to keepFit0, new to nokeepFit0
		// only first popLen individuals are kept
		for(int i = 0; i < popLen; i++) {
			if(index0[2 * popLen -1 - i] < popLen) {
				keepFit0.add(index0[2 * popLen -1 - i]);
			}else{
				nokeepFit0.add(index0[2 * popLen -1 - i]);
			}
		}

		boolean flag = false;
		//update the pbest0, if some new generated are good
		if(!nokeepFit0.isEmpty()) {
			for(int i = 0; i < popLen; i++) {
				if(!keepFit0.contains(i)) {
					//System.out.println(nokeepFit0.get(0) - popLen);
					pbest0[i] = posM[nokeepFit0.get(0) - popLen].clone();
					pbestFit0[i] = wFit0[nokeepFit0.get(0)];
					nokeepFit0.remove(0);
					if(!flag) {
						bestIndex = i;
						flag = true;
					}
				}
			}
		}
		//get the returned index
		if(index0[2 * popLen -1] < popLen) {
			bestIndex = -1;
		}
		return bestIndex;

	}
	
	// assisted step for hillvaley
	
	
	private Instances PKIdisc(Instances data) {
		// use PKI discrete method to discrete data,
		Instances dataDiscrete = null;
		Instances dataCopy = new Instances(data);
		PKIDiscretize discrete = new PKIDiscretize();
		
		try {
			discrete.setInputFormat(dataCopy);
			dataDiscrete = Filter.useFilter(dataCopy, discrete);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return dataDiscrete;
		
		
	}
	
	private void getweightForIG(String iniMethod, double[] score) throws Exception {
	
		AttributeEvaluator eval = null;
		Instances dataD = PKIdisc(this.data); 
		switch(iniMethod.toLowerCase()) {
		case "relieff":
			eval = new ReliefFAttributeEval();
			((ReliefFAttributeEval) eval).buildEvaluator(dataD);
			System.out.println("relieff");
			break;
		case "su":
			eval = new SymmetricalUncertAttributeEval();
			((SymmetricalUncertAttributeEval) eval).buildEvaluator(dataD);
			break;
		case "ig":
			eval = new InfoGainAttributeEval();
			((InfoGainAttributeEval) eval).buildEvaluator(dataD);
			System.out.println("ig");
			break;
		}
		for(int i = 0; i < fNum; i++) {
			score[i] = eval.evaluateAttribute(i);

		}
		
		double tempNormScore;
		double maxScore = score[weka.core.Utils.maxIndex(score)];
		double minScore = score[weka.core.Utils.minIndex(score)];
		double scale = 2 * sigPara[2] / (maxScore - minScore);
		for(int i = 0; i < fNum; i++) {
			tempNormScore = -sigPara[2] + scale * (score[i] - minScore);
			score[i] = sigPara[0] + (sigPara[1] - sigPara[0]) / (1.0 + Math.exp(- tempNormScore));

			//System.out.println(String.valueOf(score[i]));
		}
		
	}
	
	
	
}

