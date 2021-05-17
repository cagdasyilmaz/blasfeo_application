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

    /* free memory */
    blasfeo_free_dmat(&sA);
    v_free_align(B_mem_align);
    free(C_mem);

    /* LQ Factorization of a matrix */
    m = 13;
    n = 12;

    double *D; d_zeros(&D, n, m);
    for(ii=0; ii<n; ii++)
    {
        D[ii+n*ii] = 1.0;
        D[ii+n*(ii/2)] = 1.0;
        D[(ii/3)+n*ii] = 1.0;
        D[ii+n*(m-1)] = 1.0;
    }

    printf("\nD = \n");
    d_print_mat(n, m, D, n);

    /* matrices in blasfeo matrix struct format */
    struct blasfeo_dmat sD;
    int sD_size = blasfeo_memsize_dmat(n, m);
    void *sD_mem; v_zeros_align(&sD_mem, sD_size);
    blasfeo_create_dmat(n, m, &sD, sD_mem);
    blasfeo_pack_dmat(n, m, D, n, &sD, 0, 0);

    printf("\nsD = \n");
    blasfeo_print_dmat(n, m, &sD, 0, 0);

    /* LQ factorization */
    struct blasfeo_dmat sD_fact;
    void *sD_fact_mem; v_zeros_align(&sD_fact_mem, sD_size);
    blasfeo_create_dmat(n, m, &sD_fact, sD_fact_mem);

    int lq_size = blasfeo_dgelqf_worksize(n, m);
    void *lq_work = malloc(lq_size);

    blasfeo_dgelqf(n, m, &sD, 0, 0, &sD_fact, 0, 0, lq_work);

    printf("\nLQ fact of sD = \n");
    blasfeo_print_dmat(n, m, &sD_fact, 0, 0);
    d_print_mat(1, n, sD_fact.dA, 1);

    /* extract L */
    struct blasfeo_dmat sL;
    int sL_size = blasfeo_memsize_dmat(n, n);
    void *sL_mem; v_zeros_align(&sL_mem, sL_size);
    blasfeo_create_dmat(n, n, &sL, sL_mem);

    blasfeo_dtrcp_l(n, &sD_fact, 0, 0, &sL, 0, 0);

    printf("\nL = \n");
    blasfeo_print_dmat(n, n, &sL, 0, 0);

    /* compute Q */
    struct blasfeo_dmat sQ;
    int sQ_size = blasfeo_memsize_dmat(m, m);
    void *sQ_mem; v_zeros_align(&sQ_mem, sQ_size);
    blasfeo_create_dmat(m, m, &sQ, sQ_mem);

    int orglq_size = blasfeo_dorglq_worksize(m, m, n);
    void *orglq_work = malloc(orglq_size);

    blasfeo_dorglq(m, m, n, &sD_fact, 0, 0, &sQ, 0, 0, orglq_work);
    free(orglq_work);

    printf("\nQ = \n");
    blasfeo_print_dmat(m, m, &sQ, 0, 0);

    /* compute A-LQ */

    struct blasfeo_dmat sE;
    int sE_size = blasfeo_memsize_dmat(n, m);
    void *sE_mem; v_zeros_align(&sE_mem, sE_size);
    blasfeo_create_dmat(n, m, &sE, sE_mem);

    blasfeo_dgemm_nn(n, m, n, -1.0, &sL, 0, 0, &sQ, 0, 0, 1.0, &sD, 0, 0, &sE, 0, 0);

    printf("\nD - L * Q = \n");
    blasfeo_print_dmat(n, m, &sE, 0, 0);

    /* free memory */
    d_free(D);
    v_free_align(sD_mem);
    v_free_align(sL_mem);
    v_free_align(sQ_mem);
    v_free_align(sE_mem);
    free(lq_work);

    return 0;
}
