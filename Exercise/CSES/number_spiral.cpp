#include <iostream>
 
int main(){
    long long int n;
    std::cin >> n;
    while(n--){
        long long int x , y;
        std::cin >> x >> y;
        long long int a;
        a=x>=y?x:y;
        if(a%2==0){
            std::cout << a*a-(a-1)+x-y << std::endl;
        }else{
            std::cout << a*a-(a-1)+y-x << std::endl;
        }
    }
}
