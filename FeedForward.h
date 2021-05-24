#pragma once
#include <vector>

#include "Perceptron.h"

class FeedForward {
public:
	FeedForward(int nInputs, int nHiddenLayers, std::vector<int> nHiddenLayerNeurons, int nOutputs);
private:
	std::vector<std::vector<std::unique_ptr<Perceptron>>> neurons;
};

