
#include "device_info.h"
#include "password.h"
#include "from_list.h"
#include "common.h"
#include "bruteforce.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

int main() {
	PrintDeviceInfo();
	
	// Read passwords from tests.csv
	std::ifstream file("tests.csv");
	if (!file.is_open()) {
		std::cerr << "Error: Could not open tests.csv" << std::endl;
		return 1;
	}
	
	std::vector<std::string> passwords;
	std::string line;
	std::getline(file, line);
	std::stringstream ss(line);
	std::string password;
	
	while (std::getline(ss, password, ',')) {
		passwords.push_back(password);
	}
	
	file.close();
	
	std::cout << "\n";
	std::cout << "================================================================\n";
	std::cout << "                    PASSWORD CRACKING SUITE\n";
	std::cout << "================================================================\n\n";
	
	int testNum = 1;
	int totalSuccess = 0;
	int commonSuccess = 0, commonFail = 0;
	int listSuccess = 0, listFail = 0;
	int bruteSuccess = 0, bruteFail = 0;
	
	for (const auto& pwd : passwords) {
		std::cout << "=== Test " << testNum << ": (" << pwd << ") ===\n\n";
		
		bool found = false;
		
		// Algorithm 1: Common Passwords
		std::cout << "--- Algorithm 1: Common Passwords ---\n";
		common::Result commonResult = common::StartKernel(pwd.c_str());
		
		if (commonResult.found) {
			std::cout << "Status: SUCCESS\n";
			std::cout << "Password: " << commonResult.password << "\n";
			std::cout << "Time: " << std::fixed << std::setprecision(6) << commonResult.elapsedTime << " seconds\n";
			std::cout << "Blocks: " << commonResult.blocks << "\n";
			std::cout << "Threads per Block: " << commonResult.threadsPerBlock << "\n";
			std::cout << "Total Attempts: " << commonResult.totalAttempts << "\n";
			std::cout << "Winner Thread ID: " << commonResult.winnerThreadId << "\n";
			found = true;
			commonSuccess++;
			totalSuccess++;
		} else {
			std::cout << "Status: FAILED\n";
			std::cout << "Time: " << std::fixed << std::setprecision(6) << commonResult.elapsedTime << " seconds\n";
			commonFail++;
		}
		std::cout << "\n";
		
		// Algorithm 2: English Alphabet (Dictionary combinations)
		if (!found) {
			std::cout << "--- Algorithm 2: English Alphabet ---\n";
			from_list::Result listResult = from_list::StartKernel(pwd.c_str());
			
			if (listResult.found) {
				std::cout << "Status: SUCCESS\n";
				std::cout << "Password: " << listResult.password << "\n";
				std::cout << "Time: " << std::fixed << std::setprecision(6) << listResult.elapsedTime << " seconds\n";
				std::cout << "Blocks: " << listResult.blocksX << "x" << listResult.blocksY << "x" << listResult.blocksZ << "\n";
				std::cout << "Threads per Block: " << listResult.threadsPerBlock << "\n";
				std::cout << "Total Attempts: " << listResult.totalAttempts << "\n";
				std::cout << "Winner Thread ID: " << listResult.winnerThreadId << "\n";
				found = true;
				listSuccess++;
				totalSuccess++;
			} else {
				std::cout << "Status: FAILED\n";
				std::cout << "Time: " << std::fixed << std::setprecision(6) << listResult.elapsedTime << " seconds\n";
				listFail++;
			}
			std::cout << "\n";
		}
		
		// Algorithm 3: Bruteforce
		if (!found) {
			std::cout << "--- Algorithm 3: Bruteforce ---\n";
			brute::Result bruteResult = brute::StartKernel(pwd.c_str(), 8);
			
			if (bruteResult.found) {
				std::cout << "Status: SUCCESS\n";
				std::cout << "Password: " << bruteResult.password << "\n";
				std::cout << "Time: " << std::fixed << std::setprecision(6) << bruteResult.elapsedTime << " seconds\n";
				std::cout << "Blocks: " << bruteResult.blocksX << "x" << bruteResult.blocksY << "x" << bruteResult.blocksZ << "\n";
				std::cout << "Threads per Block: " << bruteResult.threadsPerBlock << "\n";
				std::cout << "Total Attempts: " << bruteResult.totalAttempts << "\n";
				std::cout << "Winner Thread ID: " << bruteResult.winnerThreadId << "\n";
				found = true;
				bruteSuccess++;
				totalSuccess++;
			} else {
				std::cout << "Status: FAILED\n";
				std::cout << "Time: " << std::fixed << std::setprecision(6) << bruteResult.elapsedTime << " seconds\n";
				bruteFail++;
			}
			std::cout << "\n";
		}
		
		if (!found) {
			std::cout << "*** PASSWORD NOT FOUND BY ANY ALGORITHM ***\n\n";
		}
		
		std::cout << "================================================================\n\n";
		testNum++;
	}
	
	// Summary
	std::cout << "=== SUMMARY ===\n\n";
	std::cout << "Total Tests: " << passwords.size() << "\n";
	std::cout << "Total Successes: " << totalSuccess << "\n";
	std::cout << "Total Failures: " << (passwords.size() - totalSuccess) << "\n\n";
	
	std::cout << "Common Passwords Algorithm:\n";
	std::cout << "  Successes: " << commonSuccess << "\n";
	std::cout << "  Failures: " << commonFail << "\n\n";
	
	std::cout << "English Alphabet Algorithm:\n";
	std::cout << "  Successes: " << listSuccess << "\n";
	std::cout << "  Failures: " << listFail << "\n\n";
	
	std::cout << "Bruteforce Algorithm:\n";
	std::cout << "  Successes: " << bruteSuccess << "\n";
	std::cout << "  Failures: " << bruteFail << "\n\n";
	
	std::cout << "================================================================\n";
	
	return 0;
}
