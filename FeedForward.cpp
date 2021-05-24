#include "FeedForward.h"

FeedForward::FeedForward(int nInputs, int nHiddenLayers, std::vector<int> nHiddenLayerNeurons, int nOutputs) {
	std::vector<std::unique_ptr<Perceptron>> firstLayer{};
	for (int i = 0; i < nInputs; i++) {
		firstLayer.emplace_back(std::make_unique<Perceptron>(nInputs));
	}
	neurons.emplace_back(std::move(firstLayer));
	for (int j = 0; j < nHiddenLayerNeurons.size(); j++) {
		std::vector<std::unique_ptr<Perceptron>> hiddenLayer{};
		for (int i = 0; i < nHiddenLayerNeurons[j]; i++) {
			hiddenLayer.emplace_back(std::make_unique<Perceptron>(nHiddenLayerNeurons[j]));
		}
		neurons.emplace_back(std::move(hiddenLayer));
	}
	std::vector<std::unique_ptr<Perceptron>> lastLayer{};
	for (int i = 0; i < nOutputs; i++) {
		lastLayer.emplace_back(std::make_unique<Perceptron>(nInputs));
	}
	neurons.emplace_back(std::move(lastLayer));
};

