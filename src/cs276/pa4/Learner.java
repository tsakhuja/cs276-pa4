package cs276.pa4;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Instances;

public abstract class Learner {
	String[] TFTYPES = {"url","title","body","header","anchor"};
    double smoothingBodyLength = 600;
    protected boolean usesBm25;
    protected boolean usesSmallestWindow;
    protected boolean usesPageRank;
    protected BM25Scorer bm25Scorer;
    protected SmallestWindowScorer smallestWindowScorer;

	/* Construct training features matrix */
	public abstract Instances extract_train_features(String train_data_file, String train_rel_file, Map<String,Double> idfs);

	/* Train the model */
	public abstract Classifier training (Instances dataset);
	
	/* Construct testing features matrix */
	public abstract TestFeatures extract_test_features(String test_data_file, Map<String,Double> idfs);
	
	/* Test the model, return ranked queries */
	public abstract Map<String, List<String>> testing(TestFeatures tf, Classifier model);
	
	/* Computes idf of each word in query. */
	public Map<String,Double> getQueryVector(Query q, Map<String, Double> idfs)
	{
		Map<String,Double> queryVector = new HashMap<String,Double>();
		for (String s : q.words) {
			if (idfs.containsKey(s)) {
				queryVector.put(s, idfs.get(s));
			} else {
				queryVector.put(s, idfs.get("_NONE_"));
			}
		}

		return queryVector;
	}

	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q)
	{
		// Normalize Document Vector
		for (Map<String, Double> tfMap : tfs.values()) {
			for (String t : tfMap.keySet()) {
				tfMap.put(t, tfMap.get(t) / (d.body_length + smoothingBodyLength));
			}
		}
	}


}
