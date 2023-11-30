#include <vector>
#include <iostream>
#include <numeric>

using namespace std;
#define ll long long

int main() {
	ll num, num2, num3, sum1 = 0;
	int i = 0;
	cin >> num;
	num2 = num;
	vector<ll> vec1, vec2, original_vec(num);
	while (num2 >= 1) {
		vec1.push_back(num2);
		sum1 += num2;
		if (i % 2 ==0) {
			num2 -= 3;
		} else {
			num2 -= 1;
		}
		i++;
	}
	iota(original_vec.begin(), original_vec.end(), 1);
	if (sum1 == (num + 1) * num / 4) {
		cout << "YES" << endl;
		cout << vec1.size() << endl;
		for (auto& x: vec1) {
			cout << x << " "; 
		}
		cout << endl;
		num3 = num - 1;
		i = 0;
		while (num3 >= 1) {
			vec2.push_back(num3);
			if (i % 2 ==0) {
				num3 -= 1;
			} else {
				num3 -= 3;
			}
			i++;
		}
		cout << vec2.size() << endl;
		for (auto& x: vec2) {
			cout << x << " "; 
		}
		
			
	} else {
		cout << "NO" << endl;
	}

}

