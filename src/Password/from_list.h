/// @file from_list.h
/// @author Xein
/// @date 18/11/25

#pragma once

namespace from_list {

struct Result {
	bool found;
	char password[100];
	long long winnerThreadId;
	int blocksX;
	int blocksY;
	int blocksZ;
	int threadsPerBlock;
	double elapsedTime;
	long long totalAttempts;
};

Result StartKernel(const char* target);

}
