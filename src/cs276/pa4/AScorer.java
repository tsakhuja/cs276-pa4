package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

public abstract class AScorer 
{
	
	Map<String,Double> idfs;
	String[] TFTYPES = {"url","title","body","header","anchor"};
	
	public AScorer(Map<String,Double> idfs)
	{
		this.idfs = idfs;
	}
	
	//scores each document for each query
	public abstract double getSimScore(Document d, Query q);
	
	//handle the query vector
	public Map<String,Double> getQueryFreqs(Query q)
	{
		Map<String,Double> tfQuery = new HashMap<String,Double>();
		for (String s : q.words) {
			if (tfQuery.containsKey(s)) {
				tfQuery.put(s, tfQuery.get(s) + 1.0);
			} else {
				tfQuery.put(s, 1.0);
			}
		}
		
		// Apply sublinear scaling
		for (String s : tfQuery.keySet()) {
			tfQuery.put(s, 1.0 + Math.log(tfQuery.get(s)));
		}
		
		
		return tfQuery;
	}
	

	
	////////////////////Initialization/Parsing Methods/////////////////////
	
	// URL
	String[] parseUrl(Document d) {
		return d.url.toLowerCase().split("\\W");
	}
	// Title
	String[] parseTitle(Document d){
		return d.title.toLowerCase().split("\\s+");
	}
	// Header
	String[] parseHeader(String header){
		return header.toLowerCase().split("\\s+");
	}
	String[] parseHeaders(Document d){
		List<String> headers = new ArrayList<String>();
		for (String header : d.headers) {
			String[] terms = parseHeader(header);
			for (String t : terms) {
				headers.add(t);
			}
		}
		return headers.toArray(new String[headers.size()]);
	}
	// Anchor
	String[] parseAnchor(String anchor){
		return anchor.toLowerCase().split("\\s+");
	}
	Map<String, Double> parseAnchors(Document d){
		Map<String, Double> anchors = new HashMap<String, Double>();
		for (String anchor : d.anchors.keySet()) {
			String[] terms = parseAnchor(anchor);
			for (String t : terms) {
				if (anchors.containsKey(t)) {
					anchors.put(t, anchors.get(t) + d.anchors.get(anchor));
				} else {
					anchors.put(t, (double) d.anchors.get(anchor));
				}
			}
		}
		return anchors;
	}	
	
    ////////////////////////////////////////////////////////
	
	
	/*/
	 * Creates the various kinds of term frequences (url, title, body, header, and anchor)
	 * You can override this if you'd like, but it's likely that your concrete classes will share this implementation
	 */
	public Map<String,Map<String, Double>> getDocTermFreqs(Document d, Query q)
	{
		//map from tf type -> queryWord -> score
		Map<String,Map<String, Double>> tfs = new HashMap<String,Map<String, Double>>();

		////////////////////Initialization/////////////////////
		Map<String, Double> urlTfs = new HashMap<String, Double>();
		Map<String, Double> titleTfs = new HashMap<String, Double>();
		Map<String, Double> headerTfs = new HashMap<String, Double>();
		Map<String, Double> bodyTfs = new HashMap<String, Double>();
		Map<String, Double> anchorTfs = new HashMap<String, Double>();
		for (String type : TFTYPES) {
			// URL
			if (type.equals("url") && d.url != null) {
				String[] terms = parseUrl(d);
				for (String t : terms) {
					if (q.words.contains(t)) {
						if (urlTfs.containsKey(t)) {
							urlTfs.put(t, urlTfs.get(t) + 1);
						} else {
							urlTfs.put(t, 1.0);
						}
					}
				}
			// Title
			} else if (type.equals("title") && d.title != null) {
				String[] terms = parseTitle(d);
				for (String t : terms) {
					if (q.words.contains(t)) {
						if (titleTfs.containsKey(t)) {
							titleTfs.put(t, titleTfs.get(t) + 1.0);
						} else {
							titleTfs.put(t, 1.0);
						}
					}
				}
			// Body
			} else if (type.equals("body") && d.body_hits != null) {
				for (String t : d.body_hits.keySet()) {
					t = t.toLowerCase();
					if (q.words.contains(t)) {
						bodyTfs.put(t, (double) d.body_hits.get(t).size());
					}
				}
			// Header
			} else if (type.equals("header") && d.headers != null) {
				String[] terms = parseHeaders(d);
				for (String t : terms) {
					if (q.words.contains(t)) {
						if (headerTfs.containsKey(t)) {
							headerTfs.put(t, headerTfs.get(t) + 1);
						} else {
							headerTfs.put(t, 1.0);
						}
					}
				}
			// Anchor
			} else if (type.equals("anchor") && d.anchors != null) {
				Map<String, Double> anchors = parseAnchors(d);
				for (String t : anchors.keySet()){
					if (q.words.contains(t)) {
						if (anchorTfs.containsKey(t)) {
							anchorTfs.put(t, anchorTfs.get(t) + anchors.get(t));
						} else {
							anchorTfs.put(t, (double) anchors.get(t));
						}
					}
				}
			}
		}
				
	

		
	    ////////////////////////////////////////////////////////
		
		//////////handle counts//////
		
		tfs.put("url", urlTfs);
		tfs.put("header", headerTfs);
		tfs.put("body", bodyTfs);
		tfs.put("anchor", anchorTfs);
		tfs.put("title", titleTfs);
		
		// Add sublinear scaling
		for (String type : TFTYPES) {
			for (String s : tfs.get(type).keySet()) {
				tfs.get(type).put(s, 1 + Math.log(tfs.get(type).get(s)));
			}
		}
		
		return tfs;
	}
	

}
