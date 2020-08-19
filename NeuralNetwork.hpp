/* NeuralNetwork.hpp

 From the following tutorial:
https://www.geeksforgeeks.org/ml-neural-network-implementation-in-c-from-scratch/

This hpp file defines the Neural Network Class

*/

#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>

// use typedefs for future float can be double
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

// neural network implementation case

class NeuralNetwork{
public:
	// constructor
	NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

	// forward prop function
	void propagateForward(RowVector& input);

	// neuron error backprop
	void propagateBackward(RowVector& output);

	// neuron error function for each layer
	void calcErrors(RowVector& output);

	// function to update the weights of connectors
	void updateWeights();

	// train the neural network on an array of datapoints
	void train(std::vector<RowVector*> data);

	//storage objects for class NeuralNetwork
	std::vector<RowVector*> neuronLayers;
	std::vector<RowVector*> cacheLayers;
	std::vector<RowVector*> deltas;
	std::vector<Matrix*> weights;
	Scalar learningRate;

}
