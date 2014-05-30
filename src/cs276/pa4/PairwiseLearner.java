package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
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

		Map<Query, List<Document>> trainData = null;
		Map<String, Map<String, Double>> relData = null;

		/* Load training data */
		try {
			trainData = Util.loadTrainData(train_data_file);
			relData = Util.loadRelData(train_rel_file);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		/* Construct dataset */
		Instances dataset = null;

		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		// Create list to hold nominal values
		List<String> labels = new ArrayList<String>(2); 
		labels.add("1"); 
		labels.add("-1");
		// Create nominal attribute "position" 
		Attribute label = new Attribute("label", labels);
		attributes.add(label);
		dataset = new Instances("train_dataset", attributes, 0);

		/* Add data */
		int count = 0; //Alternate classes
		for (Query query : trainData.keySet()) {
			Map<String, Double> queryVec = getQueryVector(query, idfs);
			//get all TFIDF docquery vectors
			Instances tfidfVectors = new Instances("train_dataset", attributes, 0);
			for (Document doc : trainData.get(query)) {
				Map<String,Map<String,Double>> tfs = doc.getTermFreqs();
				normalizeTFs(tfs, doc, query);
				double[] instance = new double[6];
				int j = 0;
				for (String type : TFTYPES) {
					for (String s : queryVec.keySet()) {
						if (tfs.get(type) != null && tfs.get(type).get(s) != null) {
							instance[j] += tfs.get(type).get(s) * queryVec.get(s);
						} 
					}
					j++; /* Advance to next zone */
				}
				double relScore = 0.0;
				if (relData.get(query.query) != null && relData.get(query.query).get(doc.url) != null) {
					relScore = relData.get(query.query).get(doc.url);
				}
				instance[5] = relScore;
				Instance inst = new DenseInstance(1.0, instance); 
				tfidfVectors.add(inst);
			}
			//Normalize tfidf vectors
			tfidfVectors = normalize(tfidfVectors);
			//Get pairwise vectors
			for (int i=0; i < tfidfVectors.size()-1; i++){
				for (int j=i+1; j < tfidfVectors.size(); j++){
					if (tfidfVectors.get(i).value(5) == tfidfVectors.get(j).value(5)) continue;
					//Build difference vector
					double[] nv = new double[6];
					String c;
					int l = i,r = j;//Alternate between class 1 and -1
					if (count %2 == 0){
						//Use class 1
						nv[5] = dataset.attribute(5).indexOfValue("1");
						c = "1";
						if (tfidfVectors.get(i).value(5) - tfidfVectors.get(j).value(5) < 0){
							l = j;
							r = i;
						}
					} else {
						//Use class -1
						nv[5] = dataset.attribute(5).indexOfValue("-1");
						if (tfidfVectors.get(i).value(5) - tfidfVectors.get(j).value(5) > 0){
							l = j;
							r = i;
						}
					}
					for (int k=0; i < 5; i++){
						nv[k] = tfidfVectors.get(l).value(k) - tfidfVectors.get(r).value(k);
					}
					Instance inst = new DenseInstance(1.0, nv); 
					dataset.add(inst);
					count++;
				}
			}
		}
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);

		return dataset;
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
		try {
			model.buildClassifier(dataset);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {

		Map<Query, List<Document>> testData = null;
		/* Load training data */
		try {
			testData = Util.loadTrainData(test_data_file);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		/* Construct dataset */
		Instances features = null;
		TestFeatures tf = new TestFeatures();
		int index = 0;
		Map<String, Map<String, Integer>> indexMap = new HashMap<String, Map<String, Integer>>();

		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		// Create list to hold nominal values
		List<String> labels = new ArrayList<String>(2); 
		labels.add("1"); 
		labels.add("-1");
		// Create nominal attribute "position" 
		Attribute label = new Attribute("label", labels);
		attributes.add(label);
		features = new Instances("train_dataset", attributes, 0);

		/* Add data */
		for (Query query : testData.keySet()) {
			Map<String, Integer> docIndexMap = new HashMap<String, Integer>();
			indexMap.put(query.query, docIndexMap);
			Map<String, Double> queryVec = getQueryVector(query, idfs);
			for (Document doc : testData.get(query)) {
				Map<String,Map<String,Double>> tfs = doc.getTermFreqs();
				double[] instance = new double[6];
				int i = 0;
				for (String type : TFTYPES) {
					for (String s : queryVec.keySet()) {
						if (tfs.get(type) != null && tfs.get(type).get(s) != null) {
							instance[i] += tfs.get(type).get(s) * queryVec.get(s);
						} 
					}
					i++; /* Advance to next zone */
				}
				instance[5] = 0.0;
				Instance inst = new DenseInstance(1.0, instance); 
				features.add(inst);
				docIndexMap.put(doc.url, index++);
			}
		}

		/* Set last attribute as target */
		features = normalize(features);
		tf.features = features;
		tf.index_map = indexMap;

		return tf;
	}

	@Override
	public Map<String, List<String>> testing(final TestFeatures tf,
			final Classifier model) {
		/* query -> ordered list of urls */
		Map<String, List<String>> results = new HashMap<String, List<String>>();
		for (final Entry<String, Map<String, Integer>> entry : tf.index_map.entrySet()) {
			String query = entry.getKey();
			List<String> queryResults = new ArrayList<String>();
			for (String url : entry.getValue().keySet()) {
				queryResults.add(url);
			}

			/* Sort output by weight */
			Collections.sort(queryResults, new Comparator<Object> () {
				public int compare(Object o1, Object o2) {
					Instance i1 = tf.features.get(entry.getValue().get((String) o1));
					Instance i2 = tf.features.get(entry.getValue().get((String) o2));
					try {
						return (int) (model.classifyInstance(i2) - model.classifyInstance(i1));
					} catch (Exception e) {
						return 0;
					}
				}
			});
			results.put(query, queryResults);
		}
		return results;
	}

}
