import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.concurrent.ThreadLocalRandom;

public class MGLDA {
	
	private HashMap<Integer, String> id2word;
	private LinkedList<LinkedList<LinkedList<Integer>>> doc_sentences_words;
	
	// statistics of corpus
	private int n_docs;
	private int n_of_max_sentences;
	private int n_of_max_words;
	private int vocab_size;
	private int[] doc_s_count;
    private int iter_count = 0;
	
	////////////////////////////////
	// model parameters
	////////////////////////////////
	// number of sentence covered by a sliding window, global / local topics
	private int n_windows=3;
	private int n_gl_topics = 40;
	private int n_loc_topics = 10;
	// parameter for Dirichlet prior distribution. from which we sample our global/local topics
	private double alpha_gl = 0.005;
	private double alpha_loc = 0.005; 
    // parameter for Dirichlet prior distribution. from which we sample K_GL/K_LOC word/topic distributions
    private double beta_gl= 0.005;
    private double beta_loc= 0.005;
    // parameter for Dirichlet distribution. that samples the window covering the sentence
    private double gamma = 0.005 ;
    // parameter of Beta distribution from which we sample whether a word will be assigned to a global/local
    // topic. non-symmetrical, so we can regulate whether we prefer global or local topics
    private double alpha_mix_gl = 0.005;
    private double alpha_mix_loc = 0.005;
    
    // GL_ID and LOC_ID
    private int GL_ID=1;
    private int LOC_ID=0;

    //////////////////////////////////
    // model counts
    //////////////////////////////////
    private int[][] ndk_gl;     // number of times document d and global topic k co-occur
    private int[][] ndk_loc;    // number of times document d and local topic k co-occur

    private int[] nd_gl;		// sum of document per global topic
    private int[] nd_loc;		// sum of document per local topic
    
    private int[][] nkw_gl;		// number of times word w co-occur with global topic k
    private int[][] nkw_loc;	// number of times word w co-occur with global topic k

    private int[] nk_gl; 		// sum of words per global topic
    private int[] nk_loc;		// sum of words per local topic
    
    private int [][] nds; 		// unknown
    private int [][][] ndsv; 	// unknown
    
    private int [][] ndv; 		// number of times a document d was assigned to window v
    private int [][] ndv_gl;	// number of times a global topic was assigned to d and v
    private int [][] ndv_loc;	// number of times a local topic was assigned to d and v
    private int [][][] ndvk_loc;// number of local topics in document d and window v assigned to local topic k

    private int [][][] doc_w_topics_assign;	// key is tuple of (docID, sentenceID, wordIdx), value is topic assigned
    private int [][][] doc_w_window_assign;	// key is tuple of (docID, sentenceID, wordIdx), value is window assigned
    private int [][][] doc_w_gl_loc_assign;	// key is tuple of (docID, sentenceID, wordIdx), value is gl or loc
    
	
    //////////////////////////////////
    // Distributions
    //////////////////////////////////
	private double [][] phi_dist_gl;
	private double [][] phi_dist_loc;
	
	private double [][] acc_nkw_gl;
	private double [][] acc_nkw_loc;
	
	private double [][] acc_phi_dist_gl;
	private double [][] acc_phi_dist_loc;
	
	private void println(String str) {
		System.out.println(str);
	}
	
	public MGLDA(int n_gl_toics, int n_loc_topics, HashMap<Integer, String> id2word, LinkedList<LinkedList<LinkedList<Integer>>> doc_sentences_words) {
		this.id2word = id2word;
		this.doc_sentences_words = doc_sentences_words;
		this.n_gl_topics = n_gl_toics;
		this.n_loc_topics = n_loc_topics;
		
		init_corpus_statics();
		init_counts();
	}
	
	private void init_counts() {
		this.ndk_gl = new int[this.n_docs][this.n_gl_topics];
		this.ndk_loc = new int[this.n_docs][this.n_loc_topics];
		
		this.nd_gl = new int[this.n_docs];
		this.nd_loc = new int[this.n_docs];
		
		this.nkw_gl = new int[this.n_gl_topics][this.vocab_size];
		this.nkw_loc = new int[this.n_loc_topics][this.vocab_size];
		
		this.nk_gl = new int[this.n_gl_topics];
		this.nk_loc = new int[this.n_loc_topics];
		
		this.nds = new int[this.n_docs][this.n_of_max_sentences];
		this.ndsv = new int[this.n_docs][this.n_of_max_sentences][this.n_windows];
		
		this.ndv = new int[this.n_docs][this.n_of_max_sentences + 2];
		this.ndv_gl = new int[this.n_docs][this.n_of_max_sentences + 2];
		this.ndv_loc = new int[this.n_docs][this.n_of_max_sentences + 2];
		this.ndvk_loc = new int[this.n_docs][this.n_of_max_sentences + 2][this.n_loc_topics];
		
		this.doc_w_topics_assign = new int[this.n_docs][this.n_of_max_sentences][this.n_of_max_words];
		this.doc_w_window_assign = new int[this.n_docs][this.n_of_max_sentences][this.n_of_max_words];
		this.doc_w_gl_loc_assign = new int[this.n_docs][this.n_of_max_sentences][this.n_of_max_words];
		
		this.phi_dist_gl = new double[this.n_gl_topics][this.vocab_size];
		this.phi_dist_loc = new double[this.n_loc_topics][this.vocab_size];
		
		// Accumulated topic - words count and distribution
		this.acc_nkw_gl = new double[this.n_gl_topics][this.vocab_size];
		this.acc_nkw_loc = new double[this.n_loc_topics][this.vocab_size];
		
		this.acc_phi_dist_gl = new double[this.n_gl_topics][this.vocab_size];
		this.acc_phi_dist_loc = new double[this.n_loc_topics][this.vocab_size];
	}
	
	public void init_model() {
		for (int doc_id=0; doc_id < this.n_docs; doc_id++ ) {
			int n_sents = this.doc_s_count[doc_id];
			for (int sent_id=0; sent_id < n_sents; sent_id++) {
				LinkedList<Integer> sent_words = this.doc_sentences_words.get(doc_id).get(sent_id);
				int n_words = sent_words.size();
				for (int word_id=0; word_id < n_words; word_id++) {
					int word = sent_words.get(word_id);
					
					int v = ThreadLocalRandom.current().nextInt(0, this.n_windows); // window assignment
					this.doc_w_window_assign[doc_id][sent_id][word_id] = v;
					
					int r =  ThreadLocalRandom.current().nextInt(0, 2);  // gl, loc assignment
					this.doc_w_gl_loc_assign[doc_id][sent_id][word_id] = r;
					
					this.ndv[doc_id][sent_id+v] += 1;
					this.ndsv[doc_id][sent_id][v] += 1;
					this.nds[doc_id][sent_id] += 1;
					
					int k = -1;
					if (r==this.GL_ID) {
						k = ThreadLocalRandom.current().nextInt(0, this.n_gl_topics);
						
						this.nkw_gl[k][word] += 1;
						this.ndk_gl[doc_id][k] += 1;
						this.nd_gl[doc_id] += 1;
						this.ndv_gl[doc_id][sent_id+v] += 1;
						this.nk_gl[k] += 1;
					} else {
						k = ThreadLocalRandom.current().nextInt(0, this.n_loc_topics);
						
						this.nkw_loc[k][word] += 1;
						this.ndk_loc[doc_id][k] += 1;
						this.ndv_loc[doc_id][sent_id+v] += 1;
						this.ndvk_loc[doc_id][sent_id+v][k] += 1;
						this.nd_loc[doc_id] += 1;
						this.nk_loc[k] += 1;
					}
					
					this.doc_w_topics_assign[doc_id][sent_id][word_id] = k;
				}
			}
		}
	}
	
	private void init_corpus_statics() {
		this.vocab_size = this.id2word.size();
		this.n_docs = this.doc_sentences_words.size();
		
		this.n_of_max_sentences = 0;
		this.n_of_max_words = 0;
		this.doc_s_count = new int[this.n_docs];
		for (int i=0; i < this.n_docs; i++) {
			LinkedList<LinkedList<Integer>> sent_words = doc_sentences_words.get(i);
			int n = sent_words.size();
			if (this.n_of_max_sentences <= n) {
				this.n_of_max_sentences = n;
			}
			this.doc_s_count[i] = n;
			
			for (LinkedList<Integer> words: sent_words) {
				int m = words.size();
				if (this.n_of_max_words <= m) {
					this.n_of_max_words = m;
				}
			}
		}
		
		print_corpus_statics();
	}
	
	private void lower_count(int doc_id, int sent_id, int k, int v, int r, int word) {
		this.ndv[doc_id][sent_id+v] -= 1;
		this.ndsv[doc_id][sent_id][v] -= 1;
		this.nds[doc_id][sent_id] -= 1;
		
		if (r==this.GL_ID) { // Global topic
			this.nkw_gl[k][word] -= 1;
			this.ndk_gl[doc_id][k] -= 1;
			this.nd_gl[doc_id] -= 1;
			this.ndv_gl[doc_id][sent_id+v] -= 1;
			this.nk_gl[k] -= 1;
		} else { // Local topic
			this.nkw_loc[k][word] -= 1;
			this.ndk_loc[doc_id][k] -= 1;
			this.ndv_loc[doc_id][sent_id+v] -= 1;
			this.ndvk_loc[doc_id][sent_id+v][k] -= 1;
			this.nd_loc[doc_id] -= 1;
			this.nk_loc[k] -= 1;
		}
	}
	
	private void increase_count(int doc_id, int sent_id, int k, int v, int r, int word) {
		this.ndv[doc_id][sent_id+v] += 1;
		this.ndsv[doc_id][sent_id][v] += 1;
		this.nds[doc_id][sent_id] += 1;
		
		if (r==this.GL_ID) { // Global topic
			this.nkw_gl[k][word] += 1;
			this.ndk_gl[doc_id][k] += 1;
			this.nd_gl[doc_id] += 1;
			this.ndv_gl[doc_id][sent_id+v] += 1;
			this.nk_gl[k] += 1;
		} else { // Local topic
			this.nkw_loc[k][word] += 1;
			this.ndk_loc[doc_id][k] += 1;
			this.ndv_loc[doc_id][sent_id+v] += 1;
			this.ndvk_loc[doc_id][sent_id+v][k] += 1;
			this.nd_loc[doc_id] += 1;
			this.nk_loc[k] += 1;
		}
	}
	
	private LinkedList<Integer> sample_k_v_gl_loc(int doc_id, int sent_id, int word) {
		LinkedList<Double> p_v_r_k = new LinkedList<Double>();
		LinkedList<LinkedList<Integer>> label_v_r_k = new LinkedList<LinkedList<Integer>>();
		
		for (int v=0; v < this.n_windows; v++){
			double part2 = (this.ndsv[doc_id][sent_id][v] + this.gamma) / (this.nds[doc_id][sent_id] + this.n_windows * this.gamma);
			double part3 = (this.ndv_gl[doc_id][sent_id+v] + this.alpha_mix_gl) / (this.ndv[doc_id][sent_id+v] + this.alpha_mix_gl + this.alpha_mix_loc);
			
			for (int k=0; k < this.n_gl_topics; k++) {
				LinkedList<Integer> label = new LinkedList<Integer>();
				label.add(v);
				label.add(this.GL_ID);
				label.add(k);
				label_v_r_k.add(label);
				
				double part1 = (this.nkw_gl[k][word] + this.beta_gl) / (this.nk_gl[k] + this.vocab_size * this.beta_gl);
				double part4 = (this.ndk_gl[doc_id][k] + this.alpha_gl)/ (this.nd_gl[doc_id] + this.n_gl_topics * this.alpha_gl);
				double score = part1 * part2 * part3 * part4;
				p_v_r_k.add(score);
			}
			
			part3 = (this.ndv_loc[doc_id][sent_id+v] + this.beta_loc) / (this.ndv[doc_id][sent_id+v] + this.alpha_mix_gl + this.alpha_mix_loc);
			for (int k=0; k < this.n_loc_topics; k++) {
				LinkedList<Integer> label = new LinkedList<Integer>();
				label.add(v);
				label.add(this.LOC_ID);
				label.add(k);
				label_v_r_k.add(label);
				
				double part1 = (this.nkw_loc[k][word] + this.beta_loc) / (this.nk_loc[k] + this.vocab_size * this.beta_loc);
				double part4 = (this.ndvk_loc[doc_id][sent_id+v][k] + this.alpha_loc) / (this.ndv_loc[doc_id][sent_id+v] + this.n_loc_topics * this.alpha_loc);
				double score = part1 * part2 * part3 * part4;
				p_v_r_k.add(score);
			}
		}
		
		double [] np_p_v_r_k = new double[p_v_r_k.size()];
		for (int i=0; i<p_v_r_k.size(); i++) {np_p_v_r_k[i] = p_v_r_k.get(i);}
		
		double sum = 0;
		for (double e: p_v_r_k) { sum += e;}
		for (int i=0; i<p_v_r_k.size(); i++) {np_p_v_r_k[i] /= sum;}
		
		Multinomial m = new Multinomial(np_p_v_r_k);
		int argmax = m.sample();
		
		return label_v_r_k.get(argmax);
	}

	private LinkedList<Double> calc_loglikelihood() {
		double ll_gl=0;
		double ll_loc=0;
		
		// p(w|r=gl, k)
		double[] alphas_nkw_gl = new double[this.vocab_size];
		for (int k=0; k<this.n_gl_topics; k++) {
			for (int w=0; w<this.vocab_size; w++) { alphas_nkw_gl[w] = this.nkw_gl[k][w] + this.beta_gl; }
			ll_gl += log_multi_beta(alphas_nkw_gl);
			ll_gl -= log_multi_beta(this.beta_gl, this.vocab_size);
		}
		
		// p(w|r=loc, k)
		double[] alphas_nkw_loc = new double[this.vocab_size];
		for (int k=0; k<this.n_loc_topics; k++) {
			for (int w=0; w<this.vocab_size; w++) { alphas_nkw_loc[w] = this.nkw_loc[k][w] + this.beta_loc; }
			ll_loc += log_multi_beta(alphas_nkw_loc);
			ll_loc -= log_multi_beta(this.beta_loc, this.vocab_size);
		}
		
		// p(z|r, v)
		double[] alphas_ndk_gl = new double[this.n_gl_topics];
		double[] alphas_ndk_loc = new double[this.n_loc_topics];
		for (int doc_id=0; doc_id<this.n_docs; doc_id++) {
			for (int k=0; k<this.n_gl_topics; k++) { alphas_ndk_gl[k] = this.ndk_gl[doc_id][k] + this.alpha_gl; }
			ll_gl += log_multi_beta(alphas_ndk_gl);
			ll_gl -= log_multi_beta(this.alpha_gl, this.n_gl_topics);
			
			for (int k=0; k<this.n_loc_topics; k++) { alphas_ndk_loc[k] = this.ndk_loc[doc_id][k] + this.alpha_loc; }
			ll_loc += log_multi_beta(alphas_ndk_loc);
			ll_loc -= log_multi_beta(this.alpha_loc, this.n_loc_topics);
		}
		
		// p(v|d, s)
		double[] alphas_ndv_gl = new double[this.n_of_max_sentences + 2];
		double[] alphas_ndv_loc = new double[this.n_of_max_sentences + 2];
		double[] alphas_ndsv = new double[this.n_windows];
		for (int doc_id=0; doc_id < this.n_docs; doc_id++) {
			for (int s=0; s<this.n_of_max_sentences+2; s++) { alphas_ndv_gl[s] = this.ndv_gl[doc_id][s] + this.alpha_mix_gl; }
			ll_gl += log_multi_beta(alphas_ndv_gl);
			ll_gl -= log_multi_beta(this.alpha_mix_gl, this.alpha_mix_gl + this.alpha_mix_loc);
			
			for (int s=0; s<this.n_of_max_sentences+2; s++) { alphas_ndv_loc[s] = this.ndv_loc[doc_id][s] + this.alpha_mix_loc; }
			ll_loc += log_multi_beta(alphas_ndv_loc);
			ll_loc -= log_multi_beta(this.alpha_mix_loc, this.alpha_mix_gl + this.alpha_mix_loc);
			
			for (int sent_id=0; sent_id<this.doc_s_count[doc_id]; sent_id++) {
				for (int v=0; v<this.n_windows; v++) { alphas_ndsv[v] = this.ndsv[doc_id][sent_id][v] + this.gamma; }
				ll_gl += log_multi_beta(alphas_ndsv);
				ll_gl -= log_multi_beta(this.gamma, this.n_windows);
				ll_loc += log_multi_beta(alphas_ndsv);
				ll_loc -= log_multi_beta(this.gamma, this.n_windows);
			}
		}
		
		LinkedList<Double> log_likelihood = new LinkedList<Double>();
		log_likelihood.add(ll_gl);
		log_likelihood.add(ll_loc);
		return log_likelihood;
	}

	public void run() {
		run(2);
	}
	
	public void run(int max_iter) {
		int iter = 0;
		while (iter < max_iter) {
			println("Iteration " + (this.iter_count+1));
			
			int k, v, r, word;
			int kn, vn, rn;
			LinkedList<Integer> res = null;
			for (int doc_id=0; doc_id<this.n_docs; doc_id++) {
				for (int sent_id=0; sent_id<this.doc_s_count[doc_id]; sent_id++) {
					LinkedList<Integer> words = this.doc_sentences_words.get(doc_id).get(sent_id);
					for (int word_id=0; word_id<words.size(); word_id++) {
						word = words.get(word_id);
						
						k=this.doc_w_topics_assign[doc_id][sent_id][word_id];
						v=this.doc_w_window_assign[doc_id][sent_id][word_id];
						r=this.doc_w_gl_loc_assign[doc_id][sent_id][word_id];
						
						lower_count(doc_id, sent_id, k, v, r, word);
						
						res = sample_k_v_gl_loc(doc_id, sent_id, word);
						kn = res.get(2);
						vn = res.get(0);
						rn = res.get(1);
						increase_count(doc_id, sent_id, kn, vn, rn, word);
						
						this.doc_w_topics_assign[doc_id][sent_id][word_id] = kn;
						this.doc_w_window_assign[doc_id][sent_id][word_id] = vn;
						this.doc_w_gl_loc_assign[doc_id][sent_id][word_id] = rn;
					}
				}
				
			}
			
			if (this.iter_count > 0) { build_acc_counters(); }
			
			iter++;
			this.iter_count++;
			
		}
		build_phi_matrix_gl();
		build_phi_matrix_loc();
		build_acc_phi_matrix_gl();
		build_acc_phi_matrix_loc();
	}
	
	private void build_acc_counters() {
		for (int k=0; k<this.n_gl_topics; k++) {
			for (int word_id=0; word_id<this.vocab_size; word_id++) {
				this.acc_nkw_gl[k][word_id] += (this.nkw_gl[k][word_id] + this.beta_gl);
			}
		}
		
		for (int k=0; k<this.n_loc_topics; k++) {
			for (int word_id=0; word_id<this.vocab_size; word_id++) {
				this.acc_nkw_loc[k][word_id] += this.nkw_loc[k][word_id] + this.beta_loc;
			}
		}
	}
	
	private void build_phi_matrix_gl() {
		double[][] nkw_aug = new double[this.n_gl_topics][this.vocab_size];
		
		for (int k=0; k<this.n_gl_topics; k++) {
			for(int w=0; w<this.vocab_size; w++) {
				nkw_aug[k][w] = this.nkw_gl[k][w] + this.beta_gl;
			}
		}
		
		for (int k=0; k<this.n_gl_topics; k++) {
			double num_words = 0;
			for (int word_id=0; word_id<this.vocab_size; word_id++) { num_words += nkw_aug[k][word_id]; }
			for (int word_id=0; word_id<this.vocab_size; word_id++) {
				this.phi_dist_gl[k][word_id] = nkw_aug[k][word_id] / num_words;
			}
		}
		
	}
	
	private void build_phi_matrix_loc() {
		double [][] nkw_aug = new double[this.n_loc_topics][this.vocab_size];
		
		for (int k=0; k<this.n_loc_topics; k++) {
			for(int w=0; w<this.vocab_size; w++) {
				nkw_aug[k][w] = this.nkw_loc[k][w] + this.beta_loc;
			}
		}
		
		for (int k=0; k<this.n_loc_topics; k++) {
			double num_words = 0;
			for (int word_id=0; word_id<this.vocab_size; word_id++) { num_words += nkw_aug[k][word_id]; }
			for (int word_id=0; word_id<this.vocab_size; word_id++) {
				this.phi_dist_loc[k][word_id] = nkw_aug[k][word_id] / num_words;
			}
		}
	}

	private void build_acc_phi_matrix_gl() {
		double[][] nkw_aug = new double[this.n_gl_topics][this.vocab_size];
		
		for (int k=0; k<this.n_gl_topics; k++) {
			for(int w=0; w<this.vocab_size; w++) {
				nkw_aug[k][w] = (double)(this.acc_nkw_gl[k][w]) / (this.iter_count-1) ;
			}
		}
		
		for (int k=0; k<this.n_gl_topics; k++) {
			double num_words = 0;
			for (int word_id=0; word_id<this.vocab_size; word_id++) { num_words += nkw_aug[k][word_id]; }
			for (int word_id=0; word_id<this.vocab_size; word_id++) {
				this.acc_phi_dist_gl[k][word_id] = nkw_aug[k][word_id] / num_words;
			}
		}
	}
	
	private void build_acc_phi_matrix_loc() {
		double[][] nkw_aug = new double[this.n_loc_topics][this.vocab_size];
		
		for (int k=0; k<this.n_loc_topics; k++) {
			for(int w=0; w<this.vocab_size; w++) {
				nkw_aug[k][w] = (double)(this.acc_nkw_loc[k][w]) / (this.iter_count-1) ;
			}
		}
		
		for (int k=0; k<this.n_loc_topics; k++) {
			double num_words = 0;
			for (int word_id=0; word_id<this.vocab_size; word_id++) { num_words += nkw_aug[k][word_id]; }
			for (int word_id=0; word_id<this.vocab_size; word_id++) {
				this.acc_phi_dist_loc[k][word_id] = nkw_aug[k][word_id] / num_words;
			}
		}
	}
		
	public void print_corpus_statics() {
		println("number of docs: " + this.n_docs);
		println("vocab size: " + this.vocab_size);
		println("numer of max sentences: " + this.n_of_max_sentences);
		println("numer of max words: " + this.n_of_max_words);
	}
	
	public void print_topics(BufferedWriter out) throws IOException {
		int n=30;
		
		println("Write topics to file");
		
		out.write("==========Topics==========\n");
		for (int k=0; k<this.n_gl_topics; k++) {
			out.write("========== Global topic " + k + " ==========\n");
			ArrayList<Integer> rankList = new ArrayList<Integer>();
			Utils.getTop(this.phi_dist_gl[k], rankList, n);
			
			for (int i=0; i<rankList.size(); i++) {
				int word_id = rankList.get(i);
				double prob = this.phi_dist_gl[k][word_id];
				String word = this.id2word.get(word_id);
				out.write(word + " " + prob + "\n");
			}
		}
		
		for (int k=0; k<this.n_loc_topics; k++) {
			out.write("========== Local topic " + k + " ==========\n");
			ArrayList<Integer> rankList = new ArrayList<Integer>();
			Utils.getTop(this.phi_dist_loc[k], rankList, n);
			
			for (int i=0; i<rankList.size(); i++) {
				int word_id = rankList.get(i);
				double prob = this.phi_dist_loc[k][word_id];
				String word = this.id2word.get(word_id);
				out.write(word + " " + prob + "\n");
			}
		}
	}
	
	public void save_model(String model_fname) {
		try {
			File file = new File(model_fname);
			if (!file.exists()) { file.createNewFile(); }
			BufferedWriter out = new BufferedWriter(new FileWriter(file));
			
			Utils.print_matrix(out, this.phi_dist_gl, "phi_dist_gl");
			Utils.print_matrix(out, this.phi_dist_loc, "phi_dist_loc");
			Utils.print_matrix(out, this.acc_phi_dist_gl, "acc_phi_dist_gl");
			Utils.print_matrix(out, this.acc_phi_dist_loc, "acc_phi_dist_loc");
			
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void save_checkpoint(String checkpoint_fname) {
		
		try {
			File file = new File(checkpoint_fname);
			if (!file.exists()) { file.createNewFile(); }
			BufferedWriter out = new BufferedWriter(new FileWriter(file));
			
			// assignment
			Utils.print_matrix(out, this.doc_w_window_assign, "doc_w_window_assign");
			Utils.print_matrix(out, this.doc_w_gl_loc_assign, "doc_w_gl_loc_assign");
			Utils.print_matrix(out, this.doc_w_topics_assign, "doc_w_topics_assign");
			
			// counters
			Utils.print_matrix(out, this.ndk_gl, "ndk_gl");
			Utils.print_matrix(out, this.ndk_loc, "ndk_loc");
			Utils.print_matrix(out, this.nds, "nds");
			Utils.print_matrix(out, this.ndsv, "ndsv");
			Utils.print_matrix(out, this.ndv, "ndv");
			Utils.print_matrix(out, this.ndv_gl, "ndv_gl");
			Utils.print_matrix(out, this.ndv_loc, "ndv_loc");
			Utils.print_matrix(out, this.ndvk_loc, "ndvk_loc");
			Utils.print_matrix(out, this.nkw_gl, "nkw_gl");
			Utils.print_matrix(out, this.nkw_loc, "nkw_loc");
			// Utils.print_matrix(out, this.acc_nkw_gl, "acc_nkw_gl");
			
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	}
	
	public void load_checkpoint() {
		// TODO implement this method to support continues training
	}

	private double log_multi_beta(double alpha, double K) {
		return K*Gamma.logGamma(alpha) - Gamma.logGamma(K*alpha);
	}
	
	private double log_multi_beta(double[] alphas) {
		double sum_gamma = 0;
		double sum_alpha = 0;
		
		for (double alpha: alphas) {
			sum_gamma += Gamma.logGamma(alpha);
			sum_alpha += alpha;
		}
		return sum_gamma - Gamma.logGamma(sum_alpha);
	}
}
