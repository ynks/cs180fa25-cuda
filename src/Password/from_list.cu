#include "from_list.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>

__device__ char toUpper(char c) {
	if (c >= 'a' && c <= 'z') return c - 32;
	return c;
}

__global__ void checkPasswordKernel(const char* words, int numWords, const char* target, int* found, char* result, long long* winnerThreadId) {
	// Use 3D blocks: x and y for word pairs, z for variants
	long long wordPairIdx = (long long)blockIdx.x * blockDim.x + threadIdx.x + (long long)blockIdx.y * 65535LL * blockDim.x;
	int variantIdx = blockIdx.z * blockDim.z + threadIdx.z;
	
	long long totalWordPairs = (long long)numWords * numWords;
	
	if (wordPairIdx >= totalWordPairs) return;
	if (variantIdx >= 8) return;
	if (*found) return;
	
	int word1Idx = wordPairIdx / numWords;
	int word2Idx = wordPairIdx % numWords;
	
	char password[14];
	int pos = 0;
	
	// Copy word1
	for (int i = 0; i < 6 && words[word1Idx * 6 + i] != '\0'; i++) {
		char c = words[word1Idx * 6 + i];
		// Apply capitalization for word1 based on variant
		if (variantIdx == 1 || variantIdx == 3 || variantIdx == 5 || variantIdx == 7) {
			if (i == 0) c = toUpper(c);
		}
		password[pos++] = c;
	}
	
	// Add separator
	char separator = (variantIdx < 4) ? '-' : '_';
	password[pos++] = separator;
	
	// Copy word2
	for (int i = 0; i < 6 && words[word2Idx * 6 + i] != '\0'; i++) {
		char c = words[word2Idx * 6 + i];
		// Apply capitalization for word2 based on variant
		if (variantIdx == 1 || variantIdx == 2 || variantIdx == 5 || variantIdx == 6) {
			if (i == 0) c = toUpper(c);
		}
		password[pos++] = c;
	}
	password[pos] = '\0';
	
	bool match = true;
	for (int i = 0; password[i] != '\0' && target[i] != '\0'; i++) {
		if (password[i] != target[i]) {
			match = false;
			break;
		}
	}
	if (match && password[pos] == '\0' && target[pos] == '\0') {
		if (atomicCAS(found, 0, 1) == 0) {
			for (int i = 0; i <= pos; i++) {
				result[i] = password[i];
			}
			// Calculate global thread ID
			long long globalThreadId = wordPairIdx * 8 + variantIdx;
			*winnerThreadId = globalThreadId;
		}
	}
}

void from_list::StartKernel() {
	// Load every word from "dictionary.csv"
	std::ifstream file("dictionary.csv");
	if (!file.is_open()) {
		std::cerr << "Error: Could not open dictionary.csv" << std::endl;
		return;
	}
	
	// Put it on a std::vector<std::array<char,6>> (we know max size is 6, put empty characters as \0)
	std::vector<std::array<char, 6>> words;
	std::string line;
	std::getline(file, line);
	std::stringstream ss(line);
	std::string word;
	
	while (std::getline(ss, word, ',')) {
		if (word.length() > 6) continue;
		std::array<char, 6> wordArray = {'\0'};
		for (size_t i = 0; i < word.length(); i++) {
			wordArray[i] = word[i];
		}
		words.push_back(wordArray);
	}
	
	file.close();
	
	std::cout << "Loaded " << words.size() << " words" << std::endl;
	
	// Pass that to the CUDA global memory
	char* d_words;
	cudaMalloc(&d_words, words.size() * 6);
	cudaMemcpy(d_words, words.data(), words.size() * 6, cudaMemcpyHostToDevice);
	
	const char* target = "Password-Test";
	char* d_target;
	cudaMalloc(&d_target, 12);
	cudaMemcpy(d_target, target, 12, cudaMemcpyHostToDevice);
	
	int* d_found;
	cudaMalloc(&d_found, sizeof(int));
	cudaMemset(d_found, 0, sizeof(int));
	
	char* d_result;
	cudaMalloc(&d_result, 14);
	
	long long* d_winnerThreadId;
	cudaMalloc(&d_winnerThreadId, sizeof(long long));
	cudaMemset(d_winnerThreadId, 0, sizeof(long long));
	
	// Spawn 1 thread per word1-word2-variant possibility and compare it to a target we will pass here
	int numWords = words.size();
	long long totalWordPairs = (long long)numWords * numWords;
	int threadsPerBlock = 256;
	long long blocks = (totalWordPairs + threadsPerBlock - 1) / threadsPerBlock;
	
	// Use 3D grid and block dimensions
	dim3 blockDim(threadsPerBlock, 1, 1);
	dim3 gridDim;
	if (blocks <= 65535) {
		gridDim = dim3(blocks, 1, 8);  // z-dimension for 8 variants
	} else {
		gridDim.x = 65535;
		gridDim.y = (blocks + 65534) / 65535;
		if (gridDim.y > 65535) gridDim.y = 65535;
		gridDim.z = 8;  // 8 variants in z-dimension
	}
	
	long long totalCombinations = totalWordPairs * 8;
	std::cout << "Launching kernel with " << gridDim.x << "x" << gridDim.y << "x" << gridDim.z << " blocks, " << blockDim.x << "x" << blockDim.y << "x" << blockDim.z << " threads/block" << std::endl;
	std::cout << "Total combinations: " << totalCombinations << std::endl;
	
	checkPasswordKernel<<<gridDim, blockDim>>>(d_words, numWords, d_target, d_found, d_result, d_winnerThreadId);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
	}
	
	int found;
	cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
	
	if (found) {
		char result[14];
		long long winnerThreadId;
		cudaMemcpy(result, d_result, 14, cudaMemcpyDeviceToHost);
		cudaMemcpy(&winnerThreadId, d_winnerThreadId, sizeof(long long), cudaMemcpyDeviceToHost);
		std::cout << "Password found: " << result << std::endl;
		std::cout << "Winner thread ID: " << winnerThreadId << std::endl;
	} else {
		std::cout << "Password not found" << std::endl;
	}
	
	cudaFree(d_words);
	cudaFree(d_target);
	cudaFree(d_found);
	cudaFree(d_result);
	cudaFree(d_winnerThreadId);
}

