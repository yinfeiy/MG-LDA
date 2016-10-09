
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;

public class Utils  {
	/**
	 * get a ranked list of indices from a probability distribution
	 * */
	public static void getTop(double[] array, ArrayList<Integer> rankList, int i) {
		// To implement: if i > 2*lg n, sort the whole array.
		int index = 0;
		HashSet<Integer> scanned = new HashSet<Integer>();
		double max = Float.MIN_VALUE;
		for (int m = 0; m < i && m < array.length; m++) {
			max = Float.MIN_VALUE;
			for (int no = 0; no < array.length; no++) {
				if (array[no] >= max && !scanned.contains(no)) {
					index = no;
					max = array[no];
				}
			}
			scanned.add(index);
			rankList.add(index);
			// System.out.println(m + "\t" + index);
		}
	}
	
	
	public static void print_matrix(BufferedWriter out, int[][] m, String name) throws IOException {
		int i_l, j_l;
		i_l = m.length; 
		j_l = m[0].length;
		
		out.write(name+"\n");
		out.write("" + i_l + " " + j_l + "\n");
		for (int i=0; i<m.length; i++) {
			for (int j=0; j<m[i].length; j++) {
				out.write("" + m[i][j] + " ");
			}
			out.write("\n");
		}
	}
	
	public static void print_matrix(BufferedWriter out, double[][] m, String name) throws IOException {
		int i_l, j_l;
		i_l = m.length; 
		j_l = m[0].length;
		
		out.write(name+"\n");
		out.write("" + i_l + " " + j_l + "\n");
		for (int i=0; i<m.length; i++) {
			for (int j=0; j<m[i].length; j++) {
				out.write( String.format("%.4f ", m[i][j]) );
			}
			out.write("\n");
		}
	}
	

	public static void print_matrix(BufferedWriter out, int[][][] m, String name) throws IOException {
		int i_l, j_l, k_l;
		i_l = m.length; 
		j_l = m[0].length;
		k_l = m[0][0].length;
		
		out.write(name+"\n");
		out.write("" + i_l + " " + j_l + " " + k_l + "\n");
		for (int i=0; i<m.length; i++) {
			for (int j=0; j<m[i].length; j++) {
				for (int k=0; k<m[i][j].length; k++){
					out.write("" + m[i][j][k] + " ");
				}
				out.write("\n");
			}
		}
	}
	
	
	public static void main (String[] args) {
		double[] list = {0.3, 0.2, 0.01, 0.4, 0.21};
		ArrayList<Integer> rankList = new ArrayList<Integer>();
		getTop(list, rankList, 3);
		
		System.out.println(""+rankList);
		for (int i=0;i<rankList.size(); i++) {
			System.out.println(""+list[rankList.get(i)]);
		}
	}


}
