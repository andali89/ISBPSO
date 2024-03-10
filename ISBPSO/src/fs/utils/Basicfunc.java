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
import weka.core.converters.*;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;
import weka.core.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Random;
public class Basicfunc {
	/**
	 *
	 * @param datastring: the datapath of arff data
	 * @return the data
	 */
	public static Instances readData(String datastring){
		 // read the Arffdata
		 ArffLoader read = new ArffLoader();
		 File fil = new File(datastring);
		 Instances data = null;
		 try {
			read.setFile(fil);
			data = read.getDataSet();
			data.setClassIndex(data.numAttributes()-1);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return data;
	}
	public static Instances readDataJar(String datastring){
		 // read the Arffdata
		 ArffLoader read = new ArffLoader();
		 InputStream is = read.getClass().getResourceAsStream(datastring);
		 System.out.println(is.toString());
		 Instances data = null;
		 try {
			read.setSource(is);;
			data = read.getDataSet();
			data.setClassIndex(data.numAttributes()-1);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return data;
	}
	/**
	 *
	 * @param datastring the datapath of arff data
	 * @param seed   random seed to randomize data, -1 means do not randomize
	 * @param ifnorm boolean true- means standardize to normal distribution N(0,1)
	 * @param innerfold if use inner fold to stratify data, -1 means do not stratify
	 * @return return the data
	 * @throws Exception
	 */
	public static Instances readData(String datastring, int seed, boolean ifnorm, int innerfold) throws Exception{
		// datastring,seed,ifnorm,innerfold
		Instances data = Basicfunc.readData(datastring);
		 if(seed != -1) {
			 data.randomize(new java.util.Random(seed));
		 }
		 if(innerfold > 1) {
			 data.stratify(innerfold);
		 }
		 if(ifnorm == true) {
			 data = Basicfunc.norm(data);
		 }
		 return data;
	}
	/**
	 *
	 * @param datastring the datapath of arff data
	 * @param seed   random seed to randomize data, -1 means do not randomize
	 * @param ifnorm boolean true- means standardize to normal distribution N(0,1)
	 * @param innerfold if use inner fold to stratify data, -1 means do not stratify
	 * @return return the data
	 * @throws Exception
	 */
	public static Instances readDataJar(String datastring, int seed, boolean ifnorm, int innerfold) throws Exception{
		// datastring,seed,ifnorm,innerfold
		Instances data = Basicfunc.readDataJar(datastring);
		 if(seed != -1) {
			 data.randomize(new java.util.Random(seed));
		 }
		 if(innerfold > 1) {
			 data.stratify(innerfold);
		 }
		 if(ifnorm == true) {
			 data = Basicfunc.norm(data);
		 }
		 return data;
	}
	public static Instances norm(Instances dataset) throws Exception {
		//normalize the instances of each variable
		//weka.filters.unsupervised.attribute.Standardize normer;
		//normer = new weka.filters.unsupervised.attribute.Standardize();
		Normalize normer = new Normalize();
		String scale= "2";
	    String translation = "-1";
	    normer.setOptions(new String[] {"-S",scale,"-T",translation});
	    normer.setInputFormat(dataset);
	    return weka.filters.Filter.useFilter(dataset, normer);
	}
	public static Instances standardizer(Instances dataset) throws Exception {
		//normalize the instances of each variable
		//weka.filters.unsupervised.attribute.Standardize normer;
		//normer = new weka.filters.unsupervised.attribute.Standardize();
		Standardize normer = new Standardize();		
	    normer.setInputFormat(dataset);
	    return weka.filters.Filter.useFilter(dataset, normer);
	}
	public static Instances[] divData(Instances data, int percentTest, int wholeNum, int innerFold) {
		Instances[] divSets = new Instances[2]; //Instances[0] trainset Instances[1] testset.
		data = new Instances(data);
		data.stratify(wholeNum);
		for(int i = 0; i < percentTest; i++) {
			if(i==0) {
				divSets[1] = data.testCV(wholeNum, i);
			}else {
				divSets[1].addAll(data.testCV(wholeNum, i));
			}
		}
		for(int i = percentTest; i < wholeNum; i++) {
			if(i== percentTest) {
				divSets[0] = data.testCV(wholeNum, i);
			}else {
				divSets[0].addAll(data.testCV(wholeNum, i));
			}
		}
		divSets[0].randomize(new Random(1));
		divSets[0].stratify(innerFold);
		return divSets;
	}

	/**
	 * do not need to keep class variable in subset, it is always keeped
	 * @param data
	 * @param subset in subset 1 denote retained and 0 denote eliminated
	 * @return  the filtered data of variables
	 * @throws Exception
	 */
	public static Instances select(Instances data, double[] subset) throws Exception {
		//
		//
		//
		int[] sub;
		ArrayList<Integer> list = new ArrayList<>();
		int fNum = 0;
		for(int i =  0; i < subset.length; i++) {
			if(subset[i] >= 0.5) {
				list.add(i);
				fNum += 1;
			}
		}
		list.add(subset.length);
		sub = new int[fNum + 1];
		for(int i = 0; i < fNum + 1; i++) {
			sub[i] = list.get(i);
		}
		weka.filters.unsupervised.attribute.Remove remove;
		remove = new weka.filters.unsupervised.attribute.Remove();
		remove.setAttributeIndicesArray(sub);
		remove.setInvertSelection(true);
		remove.setInputFormat(data);
		return weka.filters.Filter.useFilter(data, remove);

	}

	public static String join(String join,String[] strAry, int reNum){
		//  the first reNum elements do not join String
        StringBuffer sb=new StringBuffer();
        int strl = strAry.length;
        if(strl <= reNum) {
        	return "";        	
        }        
        for(int i=reNum;i<strl;i++){
             if(i==(strl-1)){
                 sb.append(strAry[i]);
             }else{
                 sb.append(strAry[i]).append(join);
             }
        }
        
        return new String(sb);
    }


}
