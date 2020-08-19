NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
  this->topology = topology;
  this->learningRate = learningRate;
  for (uint i = 0; i < topology.size(); i++){
    // initialize neuron layers
    if (i == topology.size() - 1)
      neuronLayers.push_back(new RowVector(topology[i]));
    else
      neuronLayers.push_back(new RowVector(topology[i] + 1));

    // initialize cache and neuron layers
    cacheLayers.push_back(new RowVector(neuronLayers.size()));
    deltas.push_back(new RowVector(neuronLayers.size()));

    /* vector.back() gives the handle to recently added element
    coeffRef gives the reference of value at that place
    implemented with pointers */

     if (i != topology.size() - 1){
       neuronLayers.back()->coeffRef(topology[i]) = 1.0;
       cacheLayers.back()->coeffRef(topology[i]) = 1.0;
     }

     //initialize weights matrix
     if (i > 0) {
       if (i != topology.size() - 1) {
         weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
         weights.back()->setRandom();
         weights.back()->col (topology[i]).setZero();
         weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
       }
       else{
         weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
         weights.back()->setRandom();
       }
     }
  }
};

// "feed forward algorithm"

void NeuralNetwork::propagateForward(RowVector& input)
{
    //set input to input layer
    // block returns part of a given vector or Matrix
    // block takes 4 arguments: startRow, startCol, blockRows, blockCols
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1)

    //forward prop implementation
    for (uint i = 1; i < topology.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
    }
    // apply activation function to NeuralNetwork
    // unaryExpr applies function to all elements of CURRENT_LAYER
    for (uint i = 1; i < topology.size() - 1; i++){
      neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(activationFunction));
    }
}

// Error calculation
void NeuralNetwork::calcErrors(RowVector& output)
{

}
