package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BM25Scorer extends AScorer
{
	Map<Query,List<Document>> queryDict;
	
	public BM25Scorer(Map<String,Double> idfs,Map<Query,List<Document>> queryDict)
	{
		super(idfs);
		this.queryDict = queryDict;
		this.calcAverageLengths();
	}

	
	///////////////weights///////////////////////////
    double urlweight = 1;
    double titleweight  = 20;
    double bodyweight = 4;
    double headerweight = 10;
    double anchorweight = 20;
    
    ///////bm25 specific weights///////////////
    double burl=1;
    double btitle=10;
    double bheader=10;
    double bbody=10;
    double banchor=10;

    double k1=50;
    double pageRankLambda=0.5;
    double pageRankLambdaPrime=20;
    
    //879

    ////////////Page rank function//////////////////////
    int pageRankFunc = 0;//0-Log 1-Inverse 2-inverse exponential
    
    
    ////////////bm25 data structures--feel free to modify ////////
    
    Map<Document,Map<String,Double>> lengths;
    Map<String,Double> avgLengths;
    Map<Document,Double> pagerankScores;
    
    //////////////////////////////////////////
    
    //sets up average lengths for bm25, also handles pagerank
    public void calcAverageLengths()
    {
    	lengths = new HashMap<Document,Map<String,Double>>();
    	avgLengths = new HashMap<String,Double>();
    	pagerankScores = new HashMap<Document,Double>();
    	
    	//get lengths for each document
    	for (Query q : queryDict.keySet()){
    		for (Document d : queryDict.get(q)){
    			Map<String,Double> docLengths = new HashMap<String, Double>();
    			for (String tfType : this.TFTYPES){
    				double l = 0;
    				if (tfType.equals("url")){
    					l = (double) parseUrl(d).length;
    				} else if (tfType.equals("title")){
    					l = (double) parseTitle(d).length;
    				} else if (tfType.equals("body")){
    					l = (double) d.body_length;
    				} else if (tfType.equals("body")){
    					l = (double) parseHeaders(d).length;
    				} else if (tfType.equals("anchors")){
        				//anchor
    					Map<String, Double> anchors = parseAnchors(d);
    					for (Double a : anchors.values()){
    						l+=a;
    					}
    				}
    				docLengths.put(tfType,l);
    				//Update overall counts
    				if (avgLengths.containsKey(tfType)){
    					avgLengths.put(tfType,avgLengths.get(tfType)+ l);
    				} else {
    					avgLengths.put(tfType,l);
    				}
    			}
    			//pagerank
    			pagerankScores.put(d, getPageRankScore(d.page_rank));
    			lengths.put(d,docLengths);
    		}

    	}
    	//normalize avgLengths
		for (String tfType : this.TFTYPES)
		{
			avgLengths.put(tfType,avgLengths.get(tfType)/lengths.keySet().size());
		}
    }

    ////////////////////////////////////
    double getPageRankScore(int pageRank){
    	switch(pageRankFunc){
    	case 1:
    		return pageRankLambdaPrime / (pageRankLambdaPrime + pageRank);
    	case 2:
    		return 1 / (pageRankLambdaPrime + Math.exp(pageRank));
    	default:
    		return Math.log(pageRankLambdaPrime + pageRank);
    	}
    }


	public double getNetScore(Map<String,Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery,Document d)
	{
		double score = 0.0;
		//For each term
		for (String term : q.words){
			//Weight each field
			double W = 0.0;
			for (String field : this.TFTYPES){
				if (tfs.get(field).containsKey(term)){
					//Get field parameter
					if (field.equals("url")){
						W += urlweight*tfs.get(field).get(term);
					} else if (field.equals("header")){
						W += headerweight*tfs.get(field).get(term);
					} else if (field.equals("body")){
						W += bodyweight*tfs.get(field).get(term);
					} else if (field.equals("title")){
						W += titleweight*tfs.get(field).get(term);
					} else if (field.equals("anchor")){
						W += anchorweight*tfs.get(field).get(term);
					}
				}
			}
			//Score the term
			score += W/(k1+W)*tfQuery.get(term)+pageRankLambda*pagerankScores.get(d);
		}
		return score;
	}

	//do bm25 normalization
	public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q)
	{
		//For each field
		for (String field : this.TFTYPES){
			//Get field parameter
			double B = 0.0;
			if (field.equals("url")){
				B = burl;
			} else if (field.equals("header")){
				B = bheader;
			} else if (field.equals("body")){
				B = bbody;
			} else if (field.equals("title")){
				B = btitle;
			} else if (field.equals("anchor")){
				B = banchor;
			}
			//For each term
			for (String term : q.words){
				//Normalize
				if (tfs.get(field).containsKey(q)){
					double denom = 1 + B*(lengths.get(d).get(field)/avgLengths.get(field)-1);
					tfs.get(field).put(term,tfs.get(field).get(term)/denom);
				}
			}
		}
	}

	
	@Override
	public double getSimScore(Document d, Query q) 
	{
		
		Map<String,Map<String, Double>> tfs = this.getDocTermFreqs(d,q);
		
		this.normalizeTFs(tfs, d, q);
		
		Map<String,Double> tfQuery = getQueryFreqs(q);
		
        return getNetScore(tfs,q,tfQuery,d);
	}

	
	
	
}
