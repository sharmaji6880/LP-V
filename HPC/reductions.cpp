#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void min_reduction(vector<int> &arr) {
    int min_value = INT_MAX;
    #pragma omp parallel for reduction(min:min_value)
    for(int i=0;i<arr.size();i++) {
        if(arr[i] < min_value) {
            min_value = arr[i];
        }
    }
    cout<<"Minimum value is:"<<min_value<<endl;
}

void max_reduction(vector<int> &arr) {
    int max_value = INT_MIN;
    #pragma omp parallel for reduction(max:max_value)
    for(int i=0;i<arr.size();i++) {
        if(arr[i] > max_value) {
            max_value = arr[i];
        }
    }
    cout<<"Maximum value is:"<<max_value<<endl;
}

void sum_reduction(vector<int> &arr) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum) 
    for(int i=0;i<arr.size();i++) {
        sum+=arr[i];
    }
    cout<<"Sum is:"<<sum<<endl;
}

void avg_reduction(vector<int> &arr) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<arr.size();i++) {
        sum+=arr[i];
    }
    cout<<"Average is:"<<(double) sum / arr.size()<<endl;
}

int main() {
    int n;
    cout<<"Enter size of array:";
    cin>>n;
    vector<int> arr(n,0);
    cout<<"Enter elements of the array:\n";
    for(int i=0;i<n;i++) {
        cin>>arr[i];
    }
    double start_time,end_time;

    start_time = omp_get_wtime();
    min_reduction(arr);
    end_time = omp_get_wtime();
    cout<<"Time taken for min reduction is:"<<end_time-start_time<<"\n\n";

    start_time = omp_get_wtime();
    max_reduction(arr);
    end_time = omp_get_wtime();
    cout<<"Time taken for max reduction is:"<<end_time-start_time<<"\n\n";

    start_time = omp_get_wtime();
    sum_reduction(arr);
    end_time = omp_get_wtime();
    cout<<"Time taken for sum reduction is:"<<end_time-start_time<<"\n\n";

    start_time = omp_get_wtime();
    avg_reduction(arr);
    end_time = omp_get_wtime();
    cout<<"Time taken for avg reduction is:"<<end_time-start_time;

    return 0;
}