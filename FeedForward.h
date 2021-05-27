#pragma once
#include <vector>

#include "Perceptron.h"

class FeedForward {
public:
	FeedForward(int nInputs, int nHiddenLayers, std::vector<int> nHiddenLayerNeurons, int nOutputs);

	template <typename T>
	void train(
		std::map<
		const VectorXd,
		double,
		T,
		Eigen::aligned_allocator<std::pair<const VectorXd, double>>
		> inOutPairs
	) {
		VectorXd outputs();
		for(auto [key, target] : inOutPairs) {
			outputs = (key);
			for(auto& layer : neurons) {
				for(auto& n : layer) {
					n->getOutput(outputs);
				}
			}
		}
		
	};

private:
	std::vector<std::vector<std::unique_ptr<Perceptron>>> neurons;
};

