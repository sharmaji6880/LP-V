#include <iostream>
#include <omp.h>
#include <vector>
#include <ctime>
using namespace std;

vector<int> sequentialBubbleSort(vector<int> arr) {
    int n = arr.size();
    for(int i=1;i<n;i++) {
        int swapped = 0;
        for(int j=0;j<=n-1-i;j++) {
            if(arr[j] > arr[j+1]) {
                swap(arr[j],arr[j+1]);
                swapped = 1;
            }
        }
        if(!swapped) {
            break;
        }
    }
    return arr;
}

vector<int> parallelBubbleSort(vector<int> arr) {
    int n = arr.size();
    for(int i=1;i<n;i++) {
        int swapped = 0;
        #pragma omp parallel for shared(arr) reduction(||:swapped)
        for(int j=0;j<=n-1-i;j++) {
            if(arr[j] > arr[j+1]) {
                swap(arr[j],arr[j+1]);
                swapped = 1;
            }
        }
        if(!swapped) {
            break;
        }
    }
    return arr;
}

void merge(vector<int> &arr, int i1, int i2, int j1, int j2) {
    vector<int> temp(1000);
    int i = i1, j=j1, k = 0;

    while(i<=i2 && j<=j2) {
        if(arr[i] < arr[j]) {
            temp[k++] = arr[i++];
        }
        else {
            temp[k++] = arr[j++];
        }
    }
    while(i<=i2) {
        temp[k++] = arr[i++];
    }
    while(j<=j2) {
        temp[k++] = arr[j++];
    }

    for(i=i1,j=0;i<=j2;i++,j++) {
       arr[i]=temp[j];
    }   

}

void sequentialMergeSort(vector<int> &arr, int i, int j) {
    if(i < j) {
        int mid = (i+j)/2;
        sequentialMergeSort(arr,i,mid);
        sequentialMergeSort(arr,mid+1,j);
        merge(arr,i,mid,mid+1,j);
    }
}

void parallelMergeSort(vector<int> &arr, int i, int j) {
    #pragma omp parallel 
    {
        #pragma omp single 
        {
            sequentialMergeSort(arr,i,j);
        }
    }
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
    vector<int> another;
    int choice;
    do {
        cout<<"==========MENU==========\n";
        cout<<"1. Sequential Bubble Sort\n";
        cout<<"2. Parallel Bubble Sort\n";
        cout<<"3. Sequential Merge Sort\n";
        cout<<"4. Parallel Merge Sort\n";
        cout<<"5. Exit\n\n";
        cout<<"Enter choice:";
        cin>>choice;

        double start_time = omp_get_wtime();

        switch(choice) {
            case 1:
                another = sequentialBubbleSort(arr);
                break;
            case 2:
                another = parallelBubbleSort(arr);
                break;
            case 3:
                another = arr;
                sequentialMergeSort(another,0,n-1);
                break;
            case 4: 
                another = arr;
                parallelMergeSort(another,0,n-1);
                break;
            case 5:
                return 0;
            default:
                cout<<"Enter valid choice:\n";
                break;
        }

        double end_time = omp_get_wtime();


        cout<<"Sorted array is:";
        for(int i=0;i<n;i++) {
            cout<<another[i]<<" ";
        }
        cout<<"\n";
        cout<<"Time taken is:"<< end_time - start_time;
        cout<<"\n\n";


    }while(true);
    
    return 0;
}
