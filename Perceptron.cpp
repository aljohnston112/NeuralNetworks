#include "Perceptron.h"
#include "UtilNN.h"
using namespace UtilNN;

Perceptron::Perceptron(int numInputs) {
	weights.resize(numInputs + 1);
	for(int i = 0; i < numInputs; i++) {
		weights(i) = getRandomWeight( );
	}
	// Bias
	weights(numInputs) = getRandomWeight( );
};