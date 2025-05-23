#include <pthread.h>
#include <stdio.h>

void* my_function(void* arg) {
    long id = (long)arg;
    printf("Thread %ld running\n", id);
    return NULL;
}

int main() {
    pthread_t threads[3];

    for (long i = 0; i < 3; i++) {
        pthread_create(&threads[i], NULL, my_function, (void*)i);
    }

    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("All threads joined. Main done.\n");
    return 0;
}
