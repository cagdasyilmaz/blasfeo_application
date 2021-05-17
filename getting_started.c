#include <stdlib.h>
#include <stdio.h>

#include "include/blasfeo.h"

int main()
{
    // printf("Testing processor\n");

    char supportString[50];
    blasfeo_processor_library_string( supportString );
    // printf( "Library requires processor features:%s\n", supportString );

    int features = 0;
    int procCheckSucceed = blasfeo_processor_cpu_features( &features );
    blasfeo_processor_feature_string( features, supportString );
    // printf( "Processor supports features:%s\n", supportString );

    if( !procCheckSucceed )
    {
        /*
        printf("Current processor does not support the current compiled BLASFEO library.\n");
        printf("Please get a BLASFEO library compatible with this processor.\n");
        */
        exit(3);
    }

    int ii;  // loop index

    int n = 5;  // matrix size
    int m = 3;  // matrix size

    // A
    struct blasfeo_dmat sA;
    blasfeo_allocate_dmat(n, m, &sA);

    // B
    struct blasfeo_dmat sB;                       // matrix structure
    int B_size = blasfeo_memsize_dmat(m, m);      // size of memory needed by B
    void *B_mem_align;
    v_zeros_align(&B_mem_align, B_size);          // allocate memory needed by B
    blasfeo_create_dmat(m, m, &sB, B_mem_align);  // assign aligned memory to struct


    // C
    struct blasfeo_dmat sC;                                                  // matrix structure
    int C_size = blasfeo_memsize_dmat(n, m);                                 // size of memory needed by C
    C_size += 64;                                                            // 64-bytes alignment
    void *C_mem = malloc(C_size);
    void *C_mem_align = (void *) ((((unsigned long long) C_mem)+63)/64*64);  // align memory pointer
    blasfeo_create_dmat(n, m, &sC, C_mem_align);                             // assign aligned memory to struct

    // A
    double *A = malloc(n*m*sizeof(double));
    for(ii=0; ii<n*m; ii++)
        A[ii] = ii+1;
    int lda = n;
    blasfeo_pack_dmat(n, m, A, lda, &sA, 0, 0);  // convert from column-major to BLASFEO dmat
    free(A);

    // B
    blasfeo_dgese(m, m, 0.0, &sB, 0, 0);    // set B to zero
    for(ii=0; ii<m; ii++)
        BLASFEO_DMATEL(&sB, ii, ii) = 1.0;  // set B diagonal to 1.0 accessing dmat elements

    BLASFEO_DMATEL(&sB, 0, 1) = 1.0;  // set first row second column to 1.0
    BLASFEO_DMATEL(&sB, 0, 2) = 1.0;  // set first row third column to 1.0

    // C
    blasfeo_dgese(n, m, -1.0, &sC, 0, 0);  // set C to -1.0

    blasfeo_dgemm_nn(n, m, m, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sC, 0, 0, &sC, 0, 0);

    printf("\nA = \n");
    blasfeo_print_dmat(n, m, &sA, 0, 0);

    printf("\nB = \n");
    blasfeo_print_dmat(m, m, &sB, 0, 0);

    printf("\nC = A * B \n");
    blasfeo_print_dmat(n, m, &sC, 0, 0);

    blasfeo_free_dmat(&sA);
    v_free_align(B_mem_align);
    free(C_mem);

    return 0;
}

