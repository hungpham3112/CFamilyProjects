#include <iostream>
#include <vector>

std::vector<int> countDecimalDigits(const std::vector<std::string>& numbers) {
    std::vector<int> decimalCounts;

    for (const std::string& number : numbers) {
        size_t decimalPointPos = number.find('.');

        // If a decimal point is found, count the characters after it
        int count = (decimalPointPos != std::string::npos) ? static_cast<int>(number.substr(decimalPointPos + 1).length()) : 0;
        decimalCounts.push_back(count);
    }

    return decimalCounts;
}

int main() {
    // Example vector of strings
    std::vector<std::string> numbers = {
        "141.573",
        "-94.4505",
        "-120.7276",
        "-13.0809",
        "128.628",
        "157.6677",
        "272.93563",
        "-226.5733",
        "-96.3777",
        "-103.2720"
    };

    // Get the vector of decimal counts
    std::vector<int> decimalCounts = countDecimalDigits(numbers);

    // Display the result
    std::cout << "Number of characters after the decimal point:\n";
    for (int count : decimalCounts) {
        std::cout << count << '\n';
    }

    return 0;
}
