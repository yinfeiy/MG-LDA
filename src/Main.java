import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Scanner;

public class Main {
	
	public static HashMap<Integer, String> parse_bow_file(String filename) {
		HashMap<Integer, String> id2word = new HashMap<Integer, String>();
		
		Scanner in = null;
		try {
			in = new Scanner(new File(filename));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		while (in.hasNext()) { // iterates each line in the file
		    String line = in.nextLine();
		    String[] ts = line.split(" ");
		    int key = Integer.parseInt(ts[0]);
		    String value = ts[1].trim();
		    id2word.put(key, value);
		}
		in.close();
		
		return id2word;	
	}
	
	public static LinkedList<LinkedList<LinkedList<Integer>>> parse_reviews(String filename) {
		LinkedList<LinkedList<LinkedList<Integer>>> doc_sentence_words = new LinkedList<LinkedList<LinkedList<Integer>>>();
		Scanner in = null;
		try {
			in = new Scanner(new File(filename));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		int doc_id_prev= -1;
		LinkedList<LinkedList<Integer>> sent_words = new LinkedList<LinkedList<Integer>>();
		while (in.hasNext()) {
			String line = in.nextLine();
			String[] ts = line.split("\t");
			int doc_id = Integer.parseInt(ts[0]);
			
			if (doc_id != doc_id_prev && doc_id_prev != -1) {
				doc_sentence_words.add(sent_words);
				sent_words = new LinkedList<LinkedList<Integer>>();
			}
			
			LinkedList<Integer> words = new LinkedList<Integer>();
			ts = ts[1].split(" ");
			for (String word: ts) {
				words.add(Integer.parseInt(word));
			}
			
			sent_words.add(words);
			doc_id_prev = doc_id;
		}
		in.close();
		return doc_sentence_words;
	}
	
	public static void main(String[] args) {
		String category = "baby";
		
		String input_dir = "./test_data/" + category + "/";
		String ifn_bow = input_dir + category + "_sample.bow";
		String ifn_inds = input_dir + category + "_sample.inds";
		
		String output_dir = "./test_results/" + category + "/";
		String ofn_topics = output_dir + category + "_sample.topics";
		String ofn_model = output_dir + category + "_sample.model";
		
		int n_gl_topics = 20;
		int n_loc_topics = 5;
		int iter_nums = 10;
		HashMap<Integer, String> id2word = parse_bow_file(ifn_bow);
		LinkedList<LinkedList<LinkedList<Integer>>> doc_sentence_words = parse_reviews(ifn_inds);
		
		MGLDA mglda = new MGLDA(n_gl_topics, n_loc_topics, id2word, doc_sentence_words);
		mglda.init_model();
		mglda.run(iter_nums);
		mglda.save_model(ofn_model);
		
		BufferedWriter out = null;
		try {
			File file = new File(ofn_topics);
			if (!file.exists()) { file.createNewFile(); }
			out = new BufferedWriter(new FileWriter(file));
			
			mglda.print_topics(out);
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
}
