import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class defaultTesting {

    public static int classIndex = -1;
	
	public static void main(String[] args) throws Exception {
		Instances above;
		Instances below;
		String[] cf;
		cf = new String[25];
		cf[0] = "0.01";
		cf[1] = "0.02";
		cf[2] = "0.03";
		cf[3] = "0.04";
		cf[4] = "0.05";
		cf[5] = "0.06";
		cf[6] = "0.07";
		cf[7] = "0.08";
		cf[8] = "0.09";
		cf[9] = "0.10";
		cf[10] = "0.11";
		cf[11] = "0.12";
		cf[12] = "0.13";
		cf[13] = "0.14";
		cf[14] = "0.15";
		cf[15] = "0.16";
		cf[16] = "0.17";
		cf[17] = "0.18";
		cf[18] = "0.19";
		cf[19] = "0.20";
		cf[20] = "0.21";
		cf[21] = "0.22";
		cf[22] = "0.23";
		cf[23] = "0.24";
		cf[24] = "0.25";
		
        DataSource source = new DataSource("docs/default2.arff");
        Instances allusers=source.getDataSet();
//        if (allusers.classIndex() == -1)
//            classIndex=allusers.numAttributes()-1;
//        allusers.setClassIndex(classIndex);
        above = new Instances(allusers,0);
        below = new Instances(allusers,0);
        
        for (int i = 0; i < allusers.numInstances(); i = i + 12) {
        	Instances user = new Instances(allusers, i, 12);
        	int counter = 0;
        	for (int j = 0; j < user.numInstances(); j++) {
        		if ((int)user.instance(j).value(user.numAttributes()-1)==0) {
					counter++;
				}
        	}
        	if (counter>=0) 
        		//System.out.println((int)allusers.instance(i).value(0)+"\t"+counter);
        		above = merge(above, user);
        	else
        		below = merge(below, user);
        }
        
        int i = 0;
        above = removeSessionID(above);
        below = removeSessionID(below);
        for (i=0; i<25; i++) {
	        Classifier fc = trainWithOption(above, cf[i]);
	        String cls = fc.toString();
	        int treeSize = Integer.parseInt( cls.substring(cls.length()-4, cls.length()-1).replaceAll(".*[^\\d](?=(\\d+))","") );
	        //double accuracy = eval(fc, allusers, allusers);
	        double accuracy = evalCrossValidation(fc, above);
	        System.out.println(cf[i]+"\t"+treeSize+"\t"+(double)Math.round(accuracy*100)/100 );
	        System.out.println(cls);
		}
//        System.out.println("===============================================================================");
//        for (i=0; i<25; i++) {
//	        Classifier fc = trainWithOption(below, cf[i]);
//	        String cls = fc.toString();
//	        int treeSize = Integer.parseInt( cls.substring(cls.length()-4, cls.length()-1).replaceAll(".*[^\\d](?=(\\d+))","") );
//	        //double accuracy = eval(fc, allusers, allusers);
//	        double accuracy = evalCrossValidation(fc, below);
//	        System.out.println(cf[i]+"\t"+treeSize+"\t"+(double)Math.round(accuracy*100)/100 );
//	        System.out.println(cls);
//		}
	}
	
	public static Instances removeSessionID(Instances pre) throws Exception {
		Instances after;
		Remove remove = new Remove();
        remove.setAttributeIndices("1");
        remove.setInputFormat(pre);
        after = Filter.useFilter(pre, remove);
		return after;
		
	}
	
    public static Classifier trainWithOption(Instances train, String cf) throws Exception
	{
    	train.setClassIndex((train.numAttributes()-1));
    	
        String[] options = new String[2];
    	options[0] = "-C";
    	options[1] = cf;
        
    	//Init classifier
    	//Classifier cls = new J48();
    	J48 j48 = new J48();
        j48.setOptions(options);
    	j48.buildClassifier(train);
    	//cls.buildClassifier(train);
        return j48;
    }
    
	public static double eval(Classifier fc, Instances train, Instances test)  throws Exception
	{
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(fc, test);
		return eval.pctCorrect();
	}
    
	public static double evalCrossValidation(Classifier cls, Instances data) throws Exception
	{
		data.setClassIndex((data.numAttributes()-1));
		Random random = new Random();
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(cls, data, 10, random);
		return eval.pctCorrect();
	}
	
    public static double selfCVEval(Instances data) throws Exception
	{
	    data.setClassIndex((data.numAttributes()-1));
	    Random random = new Random();
	    Evaluation eval = new Evaluation(data);
	    eval.crossValidateModel(new J48(), data, 10, random);
	    return eval.pctCorrect();
	}

    public static Instances merge(Instances data1, Instances data2) throws Exception
    {
        // Check where are the string attributes
        int asize = data1.numAttributes();
        boolean strings_pos[] = new boolean[asize];
        for(int i=0; i<asize; i++)
        {
            Attribute att = data1.attribute(i);
            strings_pos[i] = ((att.type() == Attribute.STRING) ||
                              (att.type() == Attribute.NOMINAL));
        }

        // Create a new dataset
        Instances dest = new Instances(data1);
//            dest.setRelationName(data1.relationName() + "+" + data2.relationName());

        DataSource source = new DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;
        while (source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            dest.add(instance);

            // Copy string attributes
            for(int i=0; i<asize; i++) {
                if(strings_pos[i]) {
                    dest.instance(dest.numInstances()-1)
                        .setValue(i,instance.stringValue(i));
                }
            }
        }
        return dest;
    }
}
