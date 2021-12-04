#pragma once
#include <random>

std::uniform_real_distribution<double> uniformRealDistribution{ 0.0, 1.0 };
std::default_random_engine defaultRandomEngine{ std::random_device{}() };

namespace UtilNN {

	double getRandomWeight( ) {
		return uniformRealDistribution(defaultRandomEngine);
	};

};