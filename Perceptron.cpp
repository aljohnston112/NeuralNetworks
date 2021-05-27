#include "Perceptron.h"
#include "UtilNN.h"
using namespace UtilNN;

Perceptron::Perceptron(
	int numInputs, 
	ActivationFunctions::ActivationFunction* activationFunctionIn,
	ErrorFunctions::ErrorFunction* errorFunctionIn,
	double learningRateIn
): activationFunction(activationFunctionIn), errorFunction(errorFunctionIn), learningRate(learningRateIn){
	weights.resize(numInputs + 1);
	for(int i = 0; i < numInputs; i++) {
		weights(i) = getRandomWeight( );
	}
	// Bias
	weights(numInputs) = getRandomWeight( );
};

