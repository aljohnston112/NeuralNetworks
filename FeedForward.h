#pragma once
#include <vector>

#include "Perceptron.h"

class FeedForward {
public:
	FeedForward(
		int nInputs,
		int nHiddenLayers,
		std::vector<int> nHiddenLayerNeurons,
		int nOutputs,
		ActivationFunctions::ActivationFunction* activationFunction,
		ErrorFunctions::ErrorFunction* errorFunction,
		double learningRate = 1
	);

	template <typename T>
	void train(
		std::map<
		const VectorXd,
		double,
		T,
		Eigen::aligned_allocator<std::pair<const VectorXd, double>>
		> inOutPairs
	) {
		VectorXd outputs(0);
		std::vector<double> newOutputs(0);
		bool good = false;
		while(!good){
			good = true;
			for(auto& [key, target] : inOutPairs) {
				outputs.resize(key.size());
				for(int i = 0; i < key.size(); i++) {
					outputs[i] = key[i];
				}
				for(auto& layer : neurons) {
					for(auto& n : layer) {
						newOutputs.emplace_back(n->getOutput(&outputs));
					}
					outputs.resize(newOutputs.size());
					for(int i = 0; i < newOutputs.size(); i++) {
						outputs[i] = newOutputs[i];
					}
					newOutputs.clear();
				}


				for(auto layerI = neurons.end()-1; layerI != neurons.begin(); --layerI){
					auto& n = *layerI;
					for(auto nI = n.end() - 1; nI != n.begin(); --nI) {
						
					}
				}


				if(outputs[0] != target) {
					good = false;
				}
			}
		}

	};

private:
	std::vector<std::vector<std::unique_ptr<Perceptron>>> neurons;

};

