package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Document {
	public String url = null;
	public String title = null;
	public List<String> headers = null;
	public Map<String, List<Integer>> body_hits = null; // term -> [list of positions]
	public int body_length = 0;
	public int page_rank = 0;
	public Map<String, Integer> anchors = null; // term -> anchor_count
	
	
	public String[] TFTYPES = {"url","title","body","header","anchor"};

	// For debug
	public String toString() {
		StringBuilder result = new StringBuilder();
		String NEW_LINE = System.getProperty("line.separator");
		if (title != null) result.append("title: " + title + NEW_LINE);
		if (headers != null) result.append("headers: " + headers.toString() + NEW_LINE);
		if (body_hits != null) result.append("body_hits: " + body_hits.toString() + NEW_LINE);
		if (body_length != 0) result.append("body_length: " + body_length + NEW_LINE);
		if (page_rank != 0) result.append("page_rank: " + page_rank + NEW_LINE);
		if (anchors != null) result.append("anchors: " + anchors.toString() + NEW_LINE);
		return result.toString();
	}
////////////////////Initialization/Parsing Methods/////////////////////
	
	// URL
	String[] parseUrl() {
		return this.url.toLowerCase().split("[:_./]+");
	}
	// Title
	String[] parseTitle(){
		return this.title.toLowerCase().split("\\s+");
	}
	// Header
	String[] parseHeader(String header){
		return header.toLowerCase().split("\\s+");
	}
	String[] parseHeaders(){
		List<String> headers = new ArrayList<String>();
		for (String header : this.headers) {
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
	Map<String, Double> parseAnchors(){
		Map<String, Double> anchors = new HashMap<String, Double>();
		for (String anchor : this.anchors.keySet()) {
			String[] terms = parseAnchor(anchor);
			for (String t : terms) {
				if (anchors.containsKey(t)) {
					anchors.put(t, anchors.get(t) + this.anchors.get(anchor));
				} else {
					anchors.put(t, (double) this.anchors.get(anchor));
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
	public Map<String,Map<String, Double>> getTermFreqs()
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
			if (type.equals("url") && this.url != null) {
				String[] terms = parseUrl();
				for (String t : terms) {
					if (urlTfs.containsKey(t)) {
						urlTfs.put(t, urlTfs.get(t) + 1);
					} else {
						urlTfs.put(t, 1.0);
					}
				}
			// Title
			} else if (type.equals("title") && this.title != null) {
				String[] terms = parseTitle();
				for (String t : terms) {
					if (titleTfs.containsKey(t)) {
						titleTfs.put(t, titleTfs.get(t) + 1.0);
					} else {
						titleTfs.put(t, 1.0);
					}
				}
			// Body
			} else if (type.equals("body") && this.body_hits != null) {
				for (String t : this.body_hits.keySet()) {
					t = t.toLowerCase();
					bodyTfs.put(t, (double) this.body_hits.get(t).size());
				}
			// Header
			} else if (type.equals("header") && this.headers != null) {
				String[] terms = parseHeaders();
				for (String t : terms) {
					if (headerTfs.containsKey(t)) {
						headerTfs.put(t, headerTfs.get(t) + 1);
					} else {
						headerTfs.put(t, 1.0);
					}
				}
			// Anchor
			} else if (type.equals("anchor") && this.anchors != null) {
				Map<String, Double> anchors = parseAnchors();
				for (String t : anchors.keySet()){
					if (anchorTfs.containsKey(t)) {
						anchorTfs.put(t, anchorTfs.get(t) + anchors.get(t));
					} else {
						anchorTfs.put(t, (double) anchors.get(t));
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
				tfs.get(type).put(s, 1.0 + Math.log(tfs.get(type).get(s)));
			}
		}
		
		return tfs;
	}
}
