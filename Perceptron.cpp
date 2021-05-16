#include "Perceptron.h"

Perceptron::Perceptron(int numInputs) {
	weights.resize(numInputs + 1);
	for(int i = 0; i < numInputs; i++) {
		weights(i) = getRandomWeight( );
	}
	// Bias
	weights(numInputs) = getRandomWeight( );
};