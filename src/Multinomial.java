import java.util.Random;

public class Multinomial {
	static Random generator = new Random();
	double[] distribution;
	int range;

	// Constructor
	Multinomial(double[] probabilities) {
		range = probabilities.length + 1;
		// We build the distribution array one larger than the array of
		// probabilities
		// to permit distribution[0] to act as a minimum bound for searching.
		// Otherwise each distribution value is a maximum bound.

		distribution = new double[range];
		double sumProb = 0;
		for (double value : probabilities) {
			sumProb += value;
		}
		distribution[0] = 0;
		for (int i = 1; i < range; ++i) {
			distribution[i] = distribution[i - 1]
					+ (probabilities[i - 1] / sumProb);
		}
		distribution[range - 1] = 1.0;
	}

	int sample() {
		// Straightforward binary search on an array of doubles to find
		// index such that distribution[i] is greater than random number while
		// distribution[i-1] is less.

		double key = generator.nextDouble();
		int mindex = 1;
		int maxdex = range - 1;
		int midpoint = mindex + (maxdex - mindex) / 2;
		while (mindex <= maxdex) {
			// System.out.println(midpoint);
			if (key < distribution[midpoint - 1]) {
				// This shouldn't ever produce an out of bounds error, since
				// it's impossible
				// that the key will be less than 0, and thus impossible that
				// the midpoint will ever be
				// zero. I think.
				maxdex = midpoint - 1;
			} else if (key > distribution[midpoint]) {
				mindex = midpoint + 1;
			} else {
				return midpoint - 1;
				// minus one, because the whole distribution array is shifted
				// one up from the
				// original probabilities array to permit distribution[0] to be
				// a minbound.
			}
			midpoint = mindex + (int) Math.ceil((maxdex - mindex) / 2);
			// I use Math.ceil to avoid any possibility of midpoint = 0.
		}
		System.out.println("Error in multinomial sampling method.");
		return range - 1;
	}
	
	public static void main(String[] args) {
		double[] probs = {0.1, 0.2, 0.7};
		Multinomial m = new Multinomial(probs);
		m.sample();
		int r = m.sample();
		System.out.println(""+r);
	}

}