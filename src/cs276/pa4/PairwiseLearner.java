package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearner extends Learner {
  private LibSVM model;
  public PairwiseLearner(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    model.setCost(C);
    model.setGamma(gamma); // only matter for RBF kernel
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
	/**	
		Map<Query, List<Document>> trainData = null;
		Map<String, Map<String, Double>> relData = null;
		
		// Load training data 
		try {
			trainData = Util.loadTrainData(train_data_file);
			relData = Util.loadRelData(train_rel_file);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// Construct dataset 
		Instances dataset = null;
		
		// Build attributes list 
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("label"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		// Add data 
		int count = 0;
		for (Query query : trainData.keySet()) {
			Map<String, Double> queryVec = null;//getQueryVector(query, idfs);
			//Compare all pairs that match query
			
			for (Document doc : trainData.get(query)) {
				 Map<String,Map<String,Double>> tfs = doc.getTermFreqs();
				 double[] instance = new double[6];
				 int i = 0;
				 for (String type : TFTYPES) {
					 for (String s : queryVec.keySet()) {
						 if (tfs.get(type) != null && tfs.get(type).get(s) != null) {
							 instance[i] += tfs.get(type).get(s) * queryVec.get(s);
						 } 
					 }
					 i++; // Advance to next zone 
				 }
				 double relScore = 0.0;
				 if (relData.get(query.query) != null && relData.get(query.query).get(doc.url) != null) {
					 relScore = relData.get(query.query).get(doc.url);
				 }
				 instance[5] = relScore;
				 Instance inst = new DenseInstance(1.0, instance); 
				 dataset.add(inst);
			}
		}
				
		// Set last attribute as target 
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		return dataset;
	}
		//Get TF-IDF Vectors from pointwise learner
		Instances tfidfVectors = null ;//PointwiseLearner.extract_train_tfidf_features(train_data_file, train_rel_file, idfs);
		
		//Normalize the vectors
		//tfidfVectors = normalize(tfidfVectors);
		
		// Build attributes list 
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("label"));
		//Initialize output vectors
		Instances dataset = new Instances("train_dataset",attributes,0);
		
		//Get relative labels
		for (int i=0; i < tfidfVectors.size()-1; i++){
			for (int j=i+1; j < tfidfVectors.size(); j++){
				double[] vi = tfidfVectors.get(i).toDoubleArray();
				double[] vj = tfidfVectors.get(j).toDoubleArray();
				if (vi[vi.length-1] == vj[vj.length-1]) continue;
				//Build difference vector
				double[] nv = Arrays.copyOf(vi,vi.length);
				for (int k=0; i < nv.length; i++){
					nv[k]-=vj[k];
				}
				//Label vector (1 of vi is ranked higher, -1 otherwise)
				if (vi[vi.length-1] - vj[vj.length-1] > 0){
					nv[nv.length-1] = 1;
				} else {
					nv[nv.length-1] = -1;
				}
				 Instance inst = new DenseInstance(1.0, nv); 
				 dataset.add(inst);
			}
		}
		return dataset;
		**/
		return null;
	}
	
	protected Instances normalize(Instances X){
		Instances X2 = null;
		try {
		Standardize filter = new Standardize();
			filter.setInputFormat(X);
			X2 = Filter.useFilter(X, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return X2;
	}

	@Override
	public Classifier training(Instances dataset) {
		/*
		 * @TODO: Your code here
		 */
		return null;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		return null;
		/**
		//Get TF-IDF Vectors from pointwise learner
		TestFeatures tfs = PointwiseLearner.extract_test_tfidf_features(test_data_file, idfs);
		Instances tfidfVectors = tfs.features;

		//Normalize the vectors
		tfidfVectors = normalize(tfidfVectors);
		
		// Build attributes list 
		// Construct dataset 
		Instances features = null;
		TestFeatures tf = new TestFeatures();
		int index = 0;
		Map<String, Map<String, Integer>> indexMap = new HashMap<String, Map<String, Integer>>();
		
		// Build attributes list 
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("label"));
		features = new Instances("test_dataset", attributes, 0);
		
		//Get relative labels
		for (int i=0; i < tfidfVectors.size()-1; i++){
			for (int j=i+1; j < tfidfVectors.size(); j++){
				double[] vi = tfidfVectors.get(i).toDoubleArray();
				double[] vj = tfidfVectors.get(j).toDoubleArray();
				if (vi[vi.length-1] == vj[vj.length-1]) continue;
				//Build difference vector
				double[] nv = Arrays.copyOf(vi,vi.length);
				for (int k=0; i < nv.length; i++){
					nv[k]-=vj[k];
				}
				//Label vector (1 of vi is ranked higher, -1 otherwise)
				if (vi[vi.length-1] - vj[vj.length-1] > 0){
					nv[nv.length-1] = 1;
				} else {
					nv[nv.length-1] = -1;
				}
				 Instance inst = new DenseInstance(1.0, nv); 
				 features.add(inst);
				 docIndexMap.put(doc.url, index++);
			}
		}
		return dataset;**/
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		/*
		 * @TODO: Your code here
		 */
		return null;
	}

}
