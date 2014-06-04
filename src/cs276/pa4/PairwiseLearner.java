package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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

	public PairwiseLearner(double C, double gamma,boolean isLinearKernel, boolean bm25, boolean window, boolean pageRank){
		try{
			model = new LibSVM();
		} catch (Exception e){
			e.printStackTrace();
		}

		model.setCost(C);
		model.setGamma(gamma); // only matter for RBF kernel
		this.usesBm25 = bm25;
		this.usesPageRank = pageRank;
		this.usesSmallestWindow = window;
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

	public ArrayList<Attribute> getAttributes(){
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("word_doc"));
		attributes.add(new Attribute("alumni"));

		if (this.usesBm25) {
			attributes.add(new Attribute("bm25_w"));
		}
		if (this.usesPageRank) {
			attributes.add(new Attribute("pagerank_w"));
		}
		if (this.usesSmallestWindow) {
			attributes.add(new Attribute("window_w"));
		}
		// Create list to hold nominal values
		List<String> labels = new ArrayList<String>(2); 
		labels.add("-1"); 
		labels.add("+1");
		// Create nominal attribute "position" 
		Attribute label = new Attribute("label", labels);
		attributes.add(label);
		return attributes;
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
		this.bm25Scorer = new BM25Scorer(idfs, trainData);
		this.smallestWindowScorer = new SmallestWindowScorer(idfs, trainData);

		/* Construct dataset */
		Instances dataset = null;

		/* Build attributes list */
		dataset = new Instances("train_dataset", getAttributes(), 0);

		/* Add data */
		int count = 0; //Alternate classes
		for (Query query : trainData.keySet()) {
			Map<String, Double> queryVec = getQueryVector(query, idfs);
			//get all TFIDF docquery vectors
			Instances tfidfVectors = new Instances("train_dataset", getAttributes(), 0);
			for (Document doc : trainData.get(query)) {
				Map<String,Map<String,Double>> tfs = doc.getTermFreqs();
				normalizeTFs(tfs, doc, query);
				double[] instance = new double[tfidfVectors.numAttributes()];
				int j = 0;
				for (String type : TFTYPES) {
					for (String s : queryVec.keySet()) {
						if (tfs.get(type) != null && tfs.get(type).get(s) != null) {
							instance[j] += tfs.get(type).get(s) * queryVec.get(s);
						} 
					}
					j++; /* Advance to next zone */
				}
				boolean containsWordDoc = doc.url.matches(".*[.](doc|ppt|xls)"); 
				instance[j++] = containsWordDoc ? 1.0 : 0.0;
				boolean containsAlumni = doc.url.matches(".*[./](news).*"); 
				instance[j++] = containsAlumni ? 1.0 : 0.0;
				if (this.usesBm25) {
					instance[j++] = this.bm25Scorer.getSimScore(doc, query);
				}
				if (this.usesPageRank) {
					instance[j++] = doc.page_rank;
				}
				if (this.usesSmallestWindow) {
					instance[j++] = this.smallestWindowScorer.getSimScore(doc, query);
				}
				double relScore = 0.0;
				if (relData.get(query.query) != null && relData.get(query.query).get(doc.url) != null) {
					relScore = relData.get(query.query).get(doc.url);
				}
				instance[tfidfVectors.numAttributes() - 1] = relScore;
				Instance inst = new DenseInstance(1.0, instance); 
				tfidfVectors.add(inst);
			}
			//Normalize tfidf vectors
			tfidfVectors = standardize(tfidfVectors);
			//Get pairwise vectors
			for (int i=0; i < tfidfVectors.size()-1; i++){
				for (int j=i+1; j < tfidfVectors.size(); j++){
					if (tfidfVectors.get(i).value(tfidfVectors.numAttributes() - 1) == tfidfVectors.get(j).value(tfidfVectors.numAttributes() - 1)) continue;
					//Build difference vector
					double[] nv1 = new double[tfidfVectors.numAttributes()];
					double[] nv2 = new double[tfidfVectors.numAttributes()];
					int l = i,r = j;//Alternate between class 1 and -1
					if (count %2 == 0){
						//Use class 1
						nv1[tfidfVectors.numAttributes() - 1] = dataset.attribute(tfidfVectors.numAttributes() - 1).indexOfValue("+1");
						nv2[tfidfVectors.numAttributes() - 1] = dataset.attribute(tfidfVectors.numAttributes() - 1).indexOfValue("-1");
						if (tfidfVectors.get(i).value(tfidfVectors.numAttributes() - 1) - tfidfVectors.get(j).value(tfidfVectors.numAttributes() - 1) < 0){
							l = j;
							r = i;
						}
					} else {
						//Use class -1
						nv1[tfidfVectors.numAttributes() - 1] = dataset.attribute(tfidfVectors.numAttributes() - 1).indexOfValue("-1");
						nv2[tfidfVectors.numAttributes() - 1] = dataset.attribute(tfidfVectors.numAttributes() - 1).indexOfValue("+1");
						if (tfidfVectors.get(i).value(tfidfVectors.numAttributes() - 1) - tfidfVectors.get(j).value(tfidfVectors.numAttributes() - 1) > 0){
							l = j;
							r = i;
						}
					}
					for (int k=0; k < tfidfVectors.numAttributes()-1; k++){
						nv1[k] = tfidfVectors.get(l).value(k) - tfidfVectors.get(r).value(k);
						nv2[k] = tfidfVectors.get(r).value(k) - tfidfVectors.get(l).value(k);
					}
					Instance inst = new DenseInstance(1.0, nv1); 
					dataset.add(inst);
					Instance inst2 = new DenseInstance(1.0, nv2); 
					dataset.add(inst2);
					
					count++;
				}
			}
		}
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);

		return dataset;
	}

	protected Instances standardize(Instances X){
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
		
		this.bm25Scorer = new BM25Scorer(idfs, testData);
		this.smallestWindowScorer = new SmallestWindowScorer(idfs,testData);

		/* Construct dataset */
		Instances features = null;
		TestFeatures tf = new TestFeatures();
		int index = 0;
		Map<String, Map<String, Integer>> indexMap = new HashMap<String, Map<String, Integer>>();

		/* Build attributes list */
		features = new Instances("test_dataset", getAttributes(), 0);

		/* Add data */
		for (Query query : testData.keySet()) {
			Map<String, Integer> docIndexMap = new HashMap<String, Integer>();
			indexMap.put(query.query, docIndexMap);
			Map<String, Double> queryVec = getQueryVector(query, idfs);
			for (Document doc : testData.get(query)) {
				Map<String,Map<String,Double>> tfs = doc.getTermFreqs();
				double[] instance = new double[features.numAttributes()];
				int i = 0;
				for (String type : TFTYPES) {
					for (String s : queryVec.keySet()) {
						if (tfs.get(type) != null && tfs.get(type).get(s) != null) {
							instance[i] += tfs.get(type).get(s) * queryVec.get(s);
						} 
					}
					i++; /* Advance to next zone */
				}
				boolean containsWordDoc = doc.url.matches(".*[.](doc|ppt|xls)"); 
				instance[i++] = containsWordDoc ? 1.0 : 0.0;
				boolean containsAlumni = doc.url.matches(".*[./](news).*"); 
				instance[i++] = containsAlumni ? 1.0 : 0.0;
				if (this.usesBm25) {
					 instance[i++] = this.bm25Scorer.getSimScore(doc, query);
				 }
				 if (this.usesPageRank) {
					 instance[i++] = doc.page_rank;
				 }
				 if (this.usesSmallestWindow) {
					 instance[i++] = this.smallestWindowScorer.getSimScore(doc, query);
				 }
				instance[features.numAttributes() - 1] = 0.0;
				Instance inst = new DenseInstance(1.0, instance); 
				features.add(inst);
				docIndexMap.put(doc.url, index++);
			}
		}

		/* Set last attribute as target */
		features.setClassIndex(features.numAttributes() - 1);
		features = standardize(features);
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
					double[] instance = new double[tf.features.numAttributes()];
					for (int i=0; i < tf.features.numAttributes()-1; i++){
						instance[i] = i1.value(i) - i2.value(i);
					}
					Instance inst = new DenseInstance(1.0,instance);
					Instances c = new Instances("comp_dataset", getAttributes(), 0);
					c.setClassIndex(c.numAttributes()-1);
					inst.setDataset(c);
					try {
						return model.classifyInstance(inst) == 0.0 ? 1 : -1;
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
