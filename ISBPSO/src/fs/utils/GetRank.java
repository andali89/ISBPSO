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

import java.util.BitSet;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.SubsetEvaluator;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.core.Instances;
import weka.core.Utils;
public class GetRank {
	
	/**
	 * 
	 * @param iniMethod
	 * @param data
	 * @return return the ranked index in descending order
	 */
	public static int[] getRankListDe(String iniMethod, Instances data) {
		int fNum = data.numAttributes() - 1;
		int[] rankList = new int[fNum]; 
		int[] tempList = new int[fNum];
		double[] score = getScore(iniMethod, data);
		tempList = Utils.sort(score);
		for(int i = 0; i < fNum; i++) {
			rankList[i] = tempList[fNum - i - 1];
		}
			
        return rankList;
     }

	/**
	 * 
	 * @param data
	 * @param ASEval
	 * @return
	 * return the ranked index in descending order, accuracy is the weight
	 */
	public static int[] getRankListDe(Instances data, ASEvaluation ASEval) {
		int fNum = data.numAttributes() - 1;
		int[] rankList = new int[fNum]; 
		int[] tempList = new int[fNum];
		double[] score = getScore(data, ASEval);
		tempList = Utils.sort(score);
		for(int i = 0; i < fNum; i++) {
			rankList[i] = tempList[fNum - i - 1];
		}
			
        return rankList;
     }
	
    /**
     * 
     * @param data
     * @param ASEval
     * @return
     * Evaluate the accuracy for each feature, and return acc as the weight
     */
	private static double[] getScore(Instances data, ASEvaluation ASEval) {
		// TODO Auto-generated method stub
		Instances dataD = data; 
		int fNum = dataD.numAttributes() - 1;
		double[] gScore = new double[fNum];
		double[] pos = null; 
		for(int i = 0; i < fNum; i++) {
			pos = Matcd.sameNums(fNum, 0);
			pos[i] = 1;
			gScore[i] = getFitness(pos, ASEval); 
		}
		return gScore;
	}

	public static double[] getScore(String iniMethod, Instances data) {

		AttributeEvaluator eval = null;
		Instances dataD = data; 
		int fNum = dataD.numAttributes() - 1;
		double[] gScore = new double[fNum];
		try {
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

		}catch (Exception e) {
			e.printStackTrace();
		}

		for(int i = 0; i < fNum; i++) {
			try {
				gScore[i] = eval.evaluateAttribute(i);
				//score[i] = 1- gScore[i];
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}

		return gScore;

	}
	
	public static double getFitness(double[] pos, ASEvaluation ASEval) {
		double acc = 0;
		int fNum = pos.length - 1;
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
	
	
	
}
