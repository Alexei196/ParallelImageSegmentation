#include<iostream>
//Pthread and UPCxx requires compatible devices, info in their documentation
//UPCxx requires specialized console line to run code. 
//#include<pthread.h>
//#include<upcxx/upcxx.hpp>

int run_pthread(char**);
int run_upcxx();
void* helloWorld(void*);

long threadMax = 4;

int main(int argc, char** argv){
    if(strcmp(argv[1], "-u") ==0 ){
        return run_upcxx();
    } else if(strcmp(argv[1], "-p") ==0 ) {
        return run_pthread();
    } else {
        std::clog << "Requires valid attribute, -u or -p"; 
        return 1;
    }
    
}

int run_pthread() {
    pthread_t* threadHandles;

    threadHandles = malloc(threadMax*sizeof(pthread_t));
    for(long i = 0; i < threadMax; ++i) {
        pthread_create(&threadHandles[i], NULL, helloWorld, (void*) (i+1));
    }

    helloWorld((void*) 0l);

    for(long i = 0; i < threadMax; ++i) {
        pthread_join(threadHandles[i], NULL);
    }

    free(threadHandles);
    return 0;
}

int run_upcxx() {
    upcxx::init();
    std::cout << "Hello from rank " << upcxx::rank_me() << " of " << upcxx::rank_n() << ".\n";
    upcxx::finalize(); 
    return 0;
}

void* helloWorld(void* rank) {
    std::cout << "Hello from rank " << (long) rank << " of " << threadMax << ".\n";
    return NULL;
}