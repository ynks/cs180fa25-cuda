/// @file common.h
/// @author Xein
/// @date 18/11/25

#pragma once

namespace common {

struct Result {
	bool found;
	char password[100];
	long long winnerThreadId;
	int blocks;
	int threadsPerBlock;
	double elapsedTime;
	int totalAttempts;
};

Result StartKernel(const char* target);

}
