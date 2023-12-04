#include <iostream>
#include <unordered_map>
#include <algorithm>


using namespace std;

int main() {
	string str;
	cin >> str;
	unordered_map<char, int> charFrequency;

	for (auto& x: str) {
		charFrequency[x]++;
	}

	int n = 0;
	string c;
	for (auto& pair: charFrequency) {
		if (pair.second % 2 == 1) {
			n++;
			c.append(pair.second, pair.first);
		}
	}
	if (n == 0) {
		for (auto& pair: charFrequency) {
			c.append(pair.second / 2, pair.first);
			reverse(c.begin(), c.end());
			c.append(pair.second / 2, pair.first);
		}
		cout << c << endl;
	} else if (n == 1) {
		for (auto& pair: charFrequency) {
			if (pair.second % 2 != 1) {
				c.append(pair.second / 2, pair.first);
				reverse(c.begin(), c.end());
				c.append(pair.second / 2, pair.first);
			}
		}
		cout << c << endl;
	}
	if (n > 1) {
		cout << "NO SOLUTION";
	}
}

