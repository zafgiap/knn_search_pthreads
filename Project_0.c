#include <stdlib.h>
#include <stdio.h>
#include "Quick_Select/quick_select.h"
#include <cblas.h>
#include "math.h"
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <openblas_config.h>

#define k 5 // k - nearest
#define N 20000 // points, like that N is replaced during linking, compiling - not dynamically, that's why we don't need malloc
#define d 10 // dimension
#define MAX_THREADS 4  
#define THREAD_MAX_POINTS 2500

float *C;
float *G;
float *distance;
float *normC2;

typedef struct{
    int N_start;
    int N_finish;  
    float *C_param;
    float *G_param;
    float *normC2_param;
} thread_args;

void VectorNorm2(float arr[], float normC2[], int N_param);
void *knn_distance(void *args);
void knn_distance_serial();

int main(void){

    srand(time(NULL));

    // Setting to 1 thread when using openblas 
    openblas_set_num_threads(1);

    C = calloc(N*d, sizeof(float));
    //G = calloc(N*N, sizeof(float));
    distance = calloc(N*N, sizeof(float));
    normC2 = calloc(N, sizeof(float));

    if ((C == NULL) || (distance == NULL) || (normC2 == NULL)){
        printf("Memory allocation failed!\n");
        exit(1);
    }

    int i=0, M=0, J=0, points_left=0, k1=0, k2=0, flag=0, iteration=0;
    struct timespec start, end;
    clock_t start_cpu, end_cpu;
    pthread_t th[8];
    
    // Deciding the number of blocks we will create 
    // || (N * d) > 100000
    if (N > THREAD_MAX_POINTS){
        
        M = N/THREAD_MAX_POINTS;
        points_left = N%THREAD_MAX_POINTS;
        flag = 1;
        iteration = 0;

    }
    else{
        M = 0;
        points_left = N;
        flag = 1;
        iteration = 0;
    }

    thread_args *args_m[MAX_THREADS];

    for (int i=0; i<MAX_THREADS; i++){
        args_m[i] = malloc(sizeof(thread_args));
    }

    // Loading the random data - from the #define's above we can have as many data as we want

    for(int i=0; i<N; i++){
        for (int j=0; j<d; j++)
            C[i*d + j] = (rand())%(200);
    }

    // matrix multiplication - needs array C and G to be float!

    // cblas_sgemm(
    //     CblasRowMajor,     
    //     CblasNoTrans,      
    //     CblasTrans,        
    //     N,                 
    //     N,                 
    //     d,                 
    //     1.0f,              
    //     C, d,              
    //     C, d,              
    //     0.0f,              
    //     G, N               
    // );

    // Calculating the norm of our array C

    VectorNorm2(C, normC2, N);

    clock_gettime(CLOCK_MONOTONIC, &start); 
    start_cpu = clock();

    // Creating threads, blocks of 10000 points

    while (flag){

        if (M >= MAX_THREADS){

            // We are setting maximum number of points to be calculated: ?

            for (int i=0; i<MAX_THREADS; i++){

                args_m[i]->N_start = i*THREAD_MAX_POINTS + (iteration*4*THREAD_MAX_POINTS);
                args_m[i]->N_finish = args_m[i]->N_start + THREAD_MAX_POINTS;
                args_m[i]->C_param = calloc((args_m[i]->N_finish -  args_m[i]->N_start) * d, sizeof(float));
                args_m[i]->G_param = calloc((args_m[i]->N_finish -  args_m[i]->N_start) * N, sizeof(float));;
                args_m[i]->normC2_param = NULL;
            
                for (int j=0; j<THREAD_MAX_POINTS; j++){
                    for (int l=0; l<d; l++)
                        args_m[i]->C_param[j*d + l] = C[(args_m[i]->N_start + j)*d + l];
                }
            
                
                if(pthread_create(&th[i], NULL, &knn_distance, (void *)args_m[i]) != 0){
                    printf("Failed to create a thread!");
                    exit(1);
                }

                printf("1 thread created!\n");

            }

            flag = 1;

        }
        else{

            flag = 0;

            for (int i=0; i<=M; i++){

                if (i < M){
                    args_m[i]->N_start = i*THREAD_MAX_POINTS + (iteration*4*THREAD_MAX_POINTS);
                    args_m[i]->N_finish = args_m[i]->N_start + THREAD_MAX_POINTS;
                    args_m[i]->C_param = calloc((args_m[i]->N_finish -  args_m[i]->N_start) * d, sizeof(float));
                    args_m[i]->G_param = calloc((args_m[i]->N_finish -  args_m[i]->N_start) * N, sizeof(float));
                    args_m[i]->normC2_param = NULL;

                    for (int j=0; j<THREAD_MAX_POINTS; j++){
                        for (int l=0; l<d; l++)
                            args_m[i]->C_param[j*d + l] = C[(args_m[i]->N_start + j)*d + l];
                    }
                
                    if (pthread_create(&th[i], NULL, &knn_distance, (void *)args_m[i]) != 0){
                        printf("Failed to create a thread!");
                        exit(1);
                    }

                    printf("1 thread created!\n");
                }
                else if(points_left > 0){
                    args_m[i]->N_start = i*THREAD_MAX_POINTS + (iteration*4*THREAD_MAX_POINTS);
                    args_m[i]->N_finish = args_m[i]->N_start + points_left;
                    args_m[i]->C_param = calloc((args_m[i]->N_finish -  args_m[i]->N_start) * d, sizeof(float));
                    args_m[i]->G_param = calloc((args_m[i]->N_finish -  args_m[i]->N_start) * N, sizeof(float));
                    args_m[i]->normC2_param = NULL;

                    for (int j=0; j<points_left; j++){
                        for (int l=0; l<d; l++)
                            args_m[i]->C_param[j*d + l] = C[(args_m[i]->N_start + j)*d + l];
                }

                    if (pthread_create(&th[i], NULL, &knn_distance, (void *)args_m[i]) != 0){
                        printf("Failed to create a thread!");
                        exit(1);
                    }

                    printf("1 thread created!\n");
                }
            
            }

        }

    

        if (M >= MAX_THREADS){

            for (int i=0; i<MAX_THREADS; i++){
                if(pthread_join(th[i], NULL) != 0){
                    printf("Failed to join a thread!");
                    exit(1);                
                }
    
            }

            // We are checking if we need to create more threads
            M -= MAX_THREADS;

            // Freeing the allocated space to C_param and G_param
            for (int i=0; i<MAX_THREADS; i++){
                if (args_m[i]->C_param != NULL)
                    free(args_m[i]->C_param);
                if (args_m[i]->G_param != NULL)
                    free(args_m[i]->G_param);
        }

        }
        else if (points_left > 0){

            for (int i=0; i<=M; i++){
                if(pthread_join(th[i], NULL) != 0){
                    printf("Failed to join a thread!");
                    exit(1);              
                }
    
            }

            for (int i=0; i<=M; i++){
                if (args_m[i]->C_param != NULL)
                    free(args_m[i]->C_param);
                if (args_m[i]->G_param != NULL)
                    free(args_m[i]->G_param);
        }

        }
        else if (M > 0){

            for (int i=0; i<M; i++){
                if(pthread_join(th[i], NULL) != 0){
                    printf("Failed to join a thread!");
                    exit(1);              
                }
    
            }
            
            for (int i=0; i<M; i++){
                if (args_m[i]->C_param != NULL)
                    free(args_m[i]->C_param);
                if (args_m[i]->G_param != NULL)
                    free(args_m[i]->G_param);
        }

        }
        else
            printf("No thread was created!\n");

        iteration++;

    }

    end_cpu = clock();
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    float elapsed_cpu = (end_cpu - start_cpu) / CLOCKS_PER_SEC;

    printf("Parallel Time running: %f seconds\n", elapsed);
    printf("Parallel CPU usage time: %f seconds\n", elapsed_cpu);


    clock_gettime(CLOCK_MONOTONIC, &start); 
    start_cpu = clock();

    // Sequensial run

    // knn_distance_serial();

    end_cpu = clock();
    clock_gettime(CLOCK_MONOTONIC, &end);

    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    elapsed_cpu = (end_cpu - start_cpu) / CLOCKS_PER_SEC;

    printf("Time running: %f seconds\n", elapsed);
    printf("CPU usage time: %f seconds\n", elapsed_cpu);

    // debug

    // for(int i=0; i<N; i++){
    //     for(int j=0; j<N; j++){
    //         printf("%f ", distance[i*N + j]);
    //     }
    //     printf("\n");
    // }

    // for(int i=0; i<N; i++){

    //     printf("Distances of Point %d (First %d distances are the %d nearest):\n", (i+1), (k+1), (k+1));

    //     for(int j=0; j<N; j++){
    //         printf("%f ", distance[i*N + j]);
    //     }
    //     printf("\n\n");
    // }

    free(C);
    free(G);
    free(distance);
    free(normC2);
    for (int i=0; i<MAX_THREADS; i++){
        free(args_m[i]);
    }


    return 0;
}   

void VectorNorm2(float arr[], float normC2[], int N_param){

    for (int i=0; i<N_param; i++){
        
        float sum = 0;

        for (int j=0; j<d; j++){

            float val = arr[i*d + j];
            sum += (val)*(val);

        }

        normC2[i] = sum;

    }

}

void *knn_distance(void *args_param){

    thread_args *args = (thread_args *)args_param;

    // Computing total G for ONLY the points of this chunk, and freeing it afterwords

    cblas_sgemm(
        CblasRowMajor,     
        CblasNoTrans,      
        CblasTrans,        
        (args->N_finish - args->N_start),                 
        N,                 
        d,                 
        1.0f,              
        args->C_param, d,              
        C, d,              
        0.0f,              
        args->G_param, N               
    );

    // calculating the distance of every thread point from every point
    for (int i=args->N_start; i<args->N_finish; i++){
        for (int j=0; j<N; j++){

            if(i!=j)
               distance[i*N + j] = sqrt(normC2[i] - 2*args->G_param[(i - args->N_start)*N + j] + normC2[j]);
            if (!(distance[i*N + j] > 0))
                distance[i*N + j] = 0;

        }
    }
    
    // maybe here we need a mutex
    // Here i is the number of points we are sorting
    
    for (int i=args->N_start; i<args->N_finish; i++){
        quickselect(distance, (i*N), (i*N)+(N-1), (i*N + k)); // relative indices, quickselect for every "row" of distances, all the distances (N in number) of a point
    }

    return NULL;

}

void knn_distance_serial(){

    // calculating the distance of every point from every point
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){

            if(i!=j)
               distance[i*N + j] = sqrt(normC2[i] - 2*G[i*N + j] + normC2[j]);
            if (!(distance[i*N + j] > 0))
                distance[i*N + j] = 0;

        }
    }

    printf("\n");

    // maybe here we need a mutex
    // Here i is the number of points we are sorting
    for (int i=0; i<N; i++){
        quickselect(distance, (i*N), (i*N)+(N-1), (i*N + k)); // relative indices, quickselect for every "row", all the distances (N in number) of a point
    }
}
