#include "quick_select.h"

int partition(float arr[], int left, int right){

    float pivot = arr[right];
    int i = left;

    for (int j = left; j < right; j++){
        if (arr[j] <= pivot){

            // if the leftmost number is less than the pivot remove it from the search (searching the proper position of pivot)
            // in the end we will have: (less than pivot)|pivot|(more than pivot)
            float temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++;
        }
    }

    float temp = arr[i];
    arr[i] = pivot;
    arr[right] = temp;

    return i; // returning the pivot position
}

int quickselect(float arr[], int left, int right, int k){

    if (left <= right){
        
        int pivotIndex = partition(arr, left, right); // left = 0, right = right - 1

        if (pivotIndex == k) // k = desiredNum - 1
            return 0;
        else if (pivotIndex > k) // we need to search again in the left section, so get rid of the right (pivotindex and everything above that)
            return quickselect(arr, left, pivotIndex - 1, k);
        else
            return quickselect(arr, pivotIndex + 1, right, k);
    }

    return -1;
}
