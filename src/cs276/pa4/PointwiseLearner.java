package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import cs276.pa4.Util;
import cs276.pa4.Document;
import cs276.pa4.Query;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PointwiseLearner extends Learner {
    
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
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		/* Add data */
		for (Query query : trainData.keySet()) {
			Map<String, Double> queryVec = getQueryVector(query, idfs);
			for (Document doc : trainData.get(query)) {
				 Map<String,Map<String,Double>> tfs = doc.getTermFreqs();
				 normalizeTFs(tfs, doc, query);
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
				 double relScore = 0.0;
				 if (relData.get(query.query) != null && relData.get(query.query).get(doc.url) != null) {
					 relScore = relData.get(query.query).get(doc.url);
				 }
				 instance[5] = relScore;
				 Instance inst = new DenseInstance(1.0, instance); 
				 dataset.add(inst);
			}
		}
				
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		return dataset;
	}
		
	@Override
	public Classifier training(Instances dataset) {
		LinearRegression model = new LinearRegression();
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
		attributes.add(new Attribute("relevance_score"));
		features = new Instances("test_dataset", attributes, 0);
		
		/* Add data */
		for (Query query : testData.keySet()) {
			Map<String, Integer> docIndexMap = new HashMap<String, Integer>();
			indexMap.put(query.query, docIndexMap);
			Map<String, Double> queryVec = getQueryVector(query, idfs);
			for (Document doc : testData.get(query)) {
				 Map<String,Map<String,Double>> tfs = doc.getTermFreqs();
				 normalizeTFs(tfs, doc, query);
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
		features.setClassIndex(features.numAttributes() - 1);
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
						return model.classifyInstance(i2) - model.classifyInstance(i1) > 0 ? 1 : -1;
					} catch (Exception e) {
						e.printStackTrace();
						return 0;
					}
				}
			});
			results.put(query, queryResults);
			
		}
		
		return results;
	}

}
