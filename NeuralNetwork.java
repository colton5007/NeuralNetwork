package io.colton5007.github.IrisClassification;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class NeuralNetwork {

	int inputLayerSize, scanOutputLayerSize, hiddenLayerSize, instances;

	double[][] x, w1, w2, y, z2, z3, a2, d2, d3, nw1, nw2, a3, scanIn, scanOut;
	double learningRate = 1;

	public static void main(String[] args) {
		
		try {
			new NeuralNetwork(Integer.parseInt(args[0]), Integer.parseInt(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]), args[4]);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * @author Colton Sandvik
	 * @param inputLayerSize
	 * Size of Input Layer Neurons or attribute count
	 * @param scanOutputLayerSize
	 * Size of scanOutput Layer of Neurons; if unsure default to 1
	 * @param hiddenlayerSize
	 * Size of Hidden Layer of Neurons. Yes there is only one; deal with it. For most problems use the average between input and scanOutput nueron count.
	 * @param instances
	 * Number of Instances that will be scanned and processed during training. More data is usually good.
	 * @param f
	 * Name of data file.
	 * @throws FileNotFoundException
	 * If you put a bad data file... you know why
	 */

	public NeuralNetwork(int inputLayerSize, int scanOutputLayerSize, int hiddenlayerSize, int instances, String f) throws FileNotFoundException {
		//Variable decleration
		this.inputLayerSize = inputLayerSize; //14
		this.scanOutputLayerSize = scanOutputLayerSize; //1
		this.hiddenLayerSize = hiddenlayerSize; //20
		this.instances = instances; //8192
		scanIn = new double[inputLayerSize][instances];
		scanOut = new double[scanOutputLayerSize][instances];
		x = new double[inputLayerSize][1];
		w1 = new double[inputLayerSize][hiddenLayerSize];
		w2 = new double[hiddenLayerSize][scanOutputLayerSize];
		y = new double[scanOutputLayerSize][1];
		z2 = new double[hiddenLayerSize][1];
		z3 = new double[scanOutputLayerSize][1];
		a3 = new double[scanOutputLayerSize][1];
		a2 = new double[hiddenLayerSize][1];
		d3 = new double[scanOutputLayerSize][1];
		d2 = new double[hiddenLayerSize][1];
		nw1 = new double[inputLayerSize][hiddenLayerSize];
		nw2 = new double[hiddenLayerSize][scanOutputLayerSize];

		//Scans file
		Scanner sc = new Scanner(new File(f));
		double[] low = new double[inputLayerSize];
		double[] high = new double[inputLayerSize];
		for (int i = 0; i < instances; i++) {
			String s = sc.nextLine();
			String[] splits = s.split(",");
			for (int j = 0; j < inputLayerSize; j++) {
				scanIn[j][i] = Double.parseDouble(splits[j]);
			}
			scanOut[0][i] = Double.parseDouble(splits[inputLayerSize]);
			//Calculates high and low for numerical normalization
			for (int j = 0; j < inputLayerSize; j++) {
				if (scanIn[j][i] < low[j])
					low[j] = scanIn[j][i];
				if (scanIn[j][i] > high[j])
					high[j] = scanIn[j][i];
			}
		}
		sc.close();
		//Numerical normalization
		for (int i = 0; i < instances; i++) {
			for (int j = 0; j < inputLayerSize; j++) {
				scanIn[j][i] = (scanIn[j][i] - low[j]) / (high[j] - low[j]);
			}
		}

		//Fills weights with random values for training
		for (int i = 0; i < inputLayerSize; i++) {
			for (int j = 0; j < hiddenLayerSize; j++) {
				w1[i][j] = Math.random() * Math.random() > 0.5 ? -1 : 1;
			}
		}
		
		for (int i = 0; i < hiddenLayerSize; i++) {
			for (int j = 0; j < scanOutputLayerSize; j++) {
				w2[i][j] = Math.random() * Math.random() > 0.5 ? -1 : 1;
			}
		}

		//Training
		//We are men, as swift as a raging river
		//We are men...
		for (int i = 0; i < instances; i++) {

			for (int j = 0; j < inputLayerSize; j++) {
				x[j][0] = scanIn[j][i];
			}
			for (int j = 0; j < scanOutputLayerSize; j++) {
				y[j][0] = scanOut[j][i];
			}
			forward();
			backward(a3);
			w1 = nw1;
			w2 = nw2;

			//Prints all the datas
			System.out.println("Case: #" + i);
			printMatrix(a3);
			printMatrix(y);

		}


	}

	/**
	 * 
	 * @param weights
	 * Matrix of weights that will be multiplied
	 * @param x
	 * Matrix of x that will be multiplied
	 * @return Matrix Product
	 * @author Colton Sandvik
	 */
	public double[][] multiplyWeights(double[][] weights, double[][] x) {
		int columns = weights.length;
		int rows = weights[0].length;
		double[][] y = new double[rows][columns];
		//Cycles through all x and multiplies each by their respective weights
		for (int i = 0; i < columns; i++) {
			for (int j = 0; j < rows; j++) {
				y[j][i] = (double) weights[i][j] * x[i][0];
			}
		}
		return y;
	}

	/**
	 * Processes Neural Network with x and y matricies. Stores final result in a3
	 */
	public void forward() {
		//Transfers x to layer 2, calculates the sum of weighted x, and runs sigmoid function over sum.
		a2 = activate(multiplyWeights(w1, x), hiddenLayerSize, 1);
		//Does the same for the next layer, pretty much.
		z3 = multiplyWeights(w2, a2);
		a3 = activate(z3, scanOutputLayerSize, 1);

	}

	/**
	 * Gets stuff done in those neurons. Activation function.
	 * @param x
	 * Weighted x that will go into a neuron
	 * @param rows
	 * Rows of matrix because I'm lazy
	 * @param columns
	 * Columns of matrix... see above
	 * @return Matrix of activated neuron data. Sum of weighted and processed with sigmoid.
	 */
	public double[][] activate(double[][] x, int rows, int columns) {
		double[][] y = new double[rows][columns];
		for (int i = 0; i < rows; i++) {
			//Calculates sum
			double sum = 0;
			for (int j = 0; j < x[i].length; j++) {
				sum += x[i][j];
			}
			//Sets z2 and z3; kinda lazy approach... but w/e
			if (rows == hiddenLayerSize) {
				z2[i][0] = sum;
			} else if (rows == scanOutputLayerSize) {
				z3[i][0] = sum;
			}
			//Sigmoid function baby
			for (int j = 0; j < columns; j++) {
				y[i][j] = sigmoid(sum);
			}
		}
		return y;
	}

	/**
	 * Implementation of sigmoid function
	 * @param z 
	 * Essentially the x of a function, but I'm using z because it doesn't look like a t and I had no other characters to pick from
	 * @return Sigmoid result
	 */
	public double sigmoid(double z) {
		return 1 / (1 + Math.pow(Math.E, -z));
	}

	/**
	 * Derivative of sigmoid function
	 * @param z
	 * Essentially the x of a function, but I'm using z because it doesn't look like a t and I had no other characters to pick from
	 * @return Slope of sigmoid at z
	 */
	public double sigmoidPrime(double z) {
		//S(z) * (1-S(z))
		return sigmoid(z) * (1 - sigmoid(z));
	}

	/**
	 * Error Cost function to determine how flawed weighted neurons are
	 * @param yHat
	 * Predicted result with current weights
	 * @param y
	 * Actual result
	 * @return Error value
	 */
	@Deprecated
	public double cost(double[][] yHat, double[][] y) {
		double sum = 0;
		for (int i = 0; i < yHat.length; i++) {
			sum += Math.pow(yHat[i][0] - y[i][0], 2);
		}
		return sum / 2;
	}

	/**
	 * Backward Propagation Algorithm
	 * @param yHat Predicted result with current weight
	 */
	public void backward(double[][] yHat) {

		//Calculates Delta for layer 3
		for (int i = 0; i < scanOutputLayerSize; i++) {
			d3[i][0] = sigmoidPrime(a3[i][0]) * (y[i][0] - yHat[i][0]);
		}

		//Uses d3 along with other "magic" a.k.a calculus to determine weights of layer 2
		for (int i = 0; i < hiddenLayerSize; i++) {
			for (int j = 0; j < scanOutputLayerSize; j++) {
				nw2[i][j] = w2[i][j] + learningRate * d3[j][0] * z2[i][0];
			}
		}

		//Calculates error for layer 2 with d3 and calculates d3
		for (int i = 0; i < hiddenLayerSize; i++) {
			d2[i][0] = sigmoidPrime(a2[i][0]);
			double e2 = 0;
			for (int j = 0; j < scanOutputLayerSize; j++) {
				e2 += d3[j][0] * w2[i][j];
			}
			d2[i][0] *= e2;
		}

		//Modifies weights of layer 1 with d2
		for (int i = 0; i < inputLayerSize; i++) {
			for (int j = 0; j < hiddenLayerSize; j++) {
				nw1[i][j] = w1[i][j] + learningRate * d2[j][0] * x[i][0];
			}
		}

	}

	/**
	 * <3 Favorite function that doesn't hurt my head
	 * @param Matrix that will be printed
	 */
	public void printMatrix(double[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println("");

	}
}
