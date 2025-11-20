/**
 * @file bruteforce.h
 * @author Dante Harper
 * @date 18/11/25
 */

#pragma once

namespace brute {

struct Result {
	bool found;
	char password[100];
	unsigned long long winnerThreadId;
	int blocksX;
	int blocksY;
	int blocksZ;
	int threadsPerBlock;
	double elapsedTime;
	unsigned long long totalAttempts;
};

Result StartKernel(const char* target, int maxLength);

}
