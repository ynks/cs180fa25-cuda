#include "common.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>

__global__ void checkCommonPasswordKernel(const char* passwords, const int* offsets, int numPasswords, const char* target, int* found, int* foundIdx, long long* winnerThreadId) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= numPasswords) return;
	if (*found) return;
	
	const char* password = passwords + offsets[idx];
	
	bool match = true;
	int i = 0;
	while (password[i] != '\0' && target[i] != '\0') {
		if (password[i] != target[i]) {
			match = false;
			break;
		}
		i++;
	}
	
	if (match && password[i] == '\0' && target[i] == '\0') {
		if (atomicCAS(found, 0, 1) == 0) {
			*foundIdx = idx;
			*winnerThreadId = (long long)blockIdx.x * blockDim.x + threadIdx.x;
		}
	}
}

common::Result common::StartKernel(const char* target) {
	Result result = {false, "", -1, 0, 0, 0.0, 0};
	auto startTime = std::chrono::high_resolution_clock::now();
	
	// Load every word from "dictionary.csv"
	std::ifstream file("most_used.csv");
	if (!file.is_open()) {
		return result;
	}

	// Put it on an std::vector<std::vector<char>>
	std::vector<std::vector<char>> passwords;
	std::string line;
	std::getline(file, line);
	std::stringstream ss(line);
	std::string word;
	
	while (std::getline(ss, word, ',')) {
		std::vector<char> p;
		p.resize(word.length() + 1);
		for (size_t i = 0; i < word.length(); i++) {
			p[i] = word[i];
		}
		p[word.length()] = '\0';
		passwords.push_back(p);
	}
	
	file.close();

	// Upload vector to CUDA memory
	std::vector<int> offsets;
	int totalSize = 0;
	for (const auto& pwd : passwords) {
		offsets.push_back(totalSize);
		totalSize += pwd.size();
	}
	
	char* flatPasswords = new char[totalSize];
	for (size_t i = 0; i < passwords.size(); i++) {
		memcpy(flatPasswords + offsets[i], passwords[i].data(), passwords[i].size());
	}
	
	char* d_passwords;
	cudaMalloc(&d_passwords, totalSize);
	cudaMemcpy(d_passwords, flatPasswords, totalSize, cudaMemcpyHostToDevice);
	
	int* d_offsets;
	cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
	cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
	
	char* d_target;
	cudaMalloc(&d_target, strlen(target) + 1);
	cudaMemcpy(d_target, target, strlen(target) + 1, cudaMemcpyHostToDevice);
	
	int* d_found;
	cudaMalloc(&d_found, sizeof(int));
	cudaMemset(d_found, 0, sizeof(int));
	
	int* d_foundIdx;
	cudaMalloc(&d_foundIdx, sizeof(int));
	
	long long* d_winnerThreadId;
	cudaMalloc(&d_winnerThreadId, sizeof(long long));
	cudaMemset(d_winnerThreadId, -1, sizeof(long long));
	
	// Spawn 1 thread per word (list is 100k most used passwords) and string compare between that and a hardcoded target i pass to every thread as a parameter const char*
	// If a thread finds it, it's selected as the winner thread and all other threads gets destroyed (similar to from_list.cu")
	int numPasswords = passwords.size();
	int threadsPerBlock = 256;
	int blocks = (numPasswords + threadsPerBlock - 1) / threadsPerBlock;
	
	result.blocks = blocks;
	result.threadsPerBlock = threadsPerBlock;
	result.totalAttempts = numPasswords;
	
	checkCommonPasswordKernel<<<blocks, threadsPerBlock>>>(d_passwords, d_offsets, numPasswords, d_target, d_found, d_foundIdx, d_winnerThreadId);
	cudaDeviceSynchronize();
	
	int found;
	cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
	
	if (found) {
		int foundIdx;
		cudaMemcpy(&foundIdx, d_foundIdx, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&result.winnerThreadId, d_winnerThreadId, sizeof(long long), cudaMemcpyDeviceToHost);
		result.found = true;
		std::string pwd(passwords[foundIdx].begin(), passwords[foundIdx].end() - 1);
		strncpy(result.password, pwd.c_str(), 99);
		result.password[99] = '\0';
	}
	
	auto endTime = std::chrono::high_resolution_clock::now();
	result.elapsedTime = std::chrono::duration<double>(endTime - startTime).count();
	
	delete[] flatPasswords;
	cudaFree(d_passwords);
	cudaFree(d_offsets);
	cudaFree(d_target);
	cudaFree(d_found);
	cudaFree(d_foundIdx);
	cudaFree(d_winnerThreadId);
	
	return result;
}
