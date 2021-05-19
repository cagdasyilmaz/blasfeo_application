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

    /* ----------- Basic Matrix Calculations and Different Matrices Declarations with Memory Allocation  ----------- */

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

    /* -------------------------------------------------------------------------------------------------------------- */

    /* ---------------------------------------- LQ Factorization of a matrix ----------------------------------------*/
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

    /* -------------------------------------------------------------------------------------------------------------- */

    /* --------------------------------------- Dense Lower-Upper Factorization -------------------------------------- */
    n = 16;

    double *F; d_zeros(&F, n, n);
    for(ii=0; ii<n*n; ii++)
        F[ii] = ii;
    //d_print_mat(n, n, F, n);

    // symmetric positive definite matrix, at the same time identity matrix
    double *G; d_zeros(&G, n, n);
    for(ii=0; ii<n; ii++)
        G[ii*(n+1)] = 1.0;
	// d_print_mat(n, n, G, n);

    // identity
    double *I; d_zeros(&I, n, n);
    for(ii=0; ii<n; ii++) I[ii*(n+1)] = 1.0;
    //d_print_mat(n, n, I, n);

    // result matrix
    double *H; d_zeros(&H, n, n);
    // d_print_mat(n, n, H, n);

    // permutation indeces
    int *ipiv; int_zeros(&ipiv, n, 1);

    /*
     * matrices in matrix struct format
     */

    // work space enough for 6 matrix structs for size n times n
    int size_strmat = 6*blasfeo_memsize_dmat(n, n);
    void *memory_strmat; v_zeros_align(&memory_strmat, size_strmat);
    char *ptr_memory_strmat = (char *) memory_strmat;

    struct blasfeo_dmat sF;
    //blasfeo_allocate_dmat(n, n, &sF);
    blasfeo_create_dmat(n, n, &sF, ptr_memory_strmat);
    ptr_memory_strmat += sF.memsize;
    // convert from column major matrix to strmat
    blasfeo_pack_dmat(n, n, F, n, &sF, 0, 0);
    printf("\nF = \n");
    blasfeo_print_dmat(n, n, &sF, 0, 0);

    struct blasfeo_dmat sG;
    //blasfeo_allocate_dmat(n, n, &sG);
    blasfeo_create_dmat(n, n, &sG, ptr_memory_strmat);
    ptr_memory_strmat += sG.memsize;
    // convert from column major matrix to strmat
    blasfeo_pack_dmat(n, n, G, n, &sG, 0, 0);
    printf("\nG = \n");
    blasfeo_print_dmat(n, n, &sG, 0, 0);

    struct blasfeo_dmat sI;
    //	blasfeo_allocate_dmat(n, n, &sI);
    blasfeo_create_dmat(n, n, &sI, ptr_memory_strmat);
    ptr_memory_strmat += sI.memsize;
    // convert from column major matrix to strmat

    struct blasfeo_dmat sH;
    //blasfeo_allocate_dmat(n, n, &sH);
    blasfeo_create_dmat(n, n, &sH, ptr_memory_strmat);
    ptr_memory_strmat += sH.memsize;

    struct blasfeo_dmat sLU;
    //blasfeo_allocate_dmat(n, n, &sD);
    blasfeo_create_dmat(n, n, &sLU, ptr_memory_strmat);
    ptr_memory_strmat += sLU.memsize;

    struct blasfeo_dmat sLUt;
    //blasfeo_allocate_dmat(n, n, &sD);
    blasfeo_create_dmat(n, n, &sLUt, ptr_memory_strmat);
    ptr_memory_strmat += sLUt.memsize;

    blasfeo_dgemm_nt(n, n, n, 1.0, &sF, 0, 0, &sF, 0, 0, 1.0, &sG, 0, 0, &sH, 0, 0);
    printf("\nG+F*F' = \n");
    blasfeo_print_dmat(n, n, &sH, 0, 0);

    //blasfeo_dgetrf_nopivot(n, n, &sH, 0, 0, &sH, 0, 0);
    blasfeo_dgetrf_rp(n, n, &sH, 0, 0, &sLU, 0, 0, ipiv);
    printf("\nLU = \n");
    blasfeo_print_dmat(n, n, &sLU, 0, 0);
    printf("\nipiv = \n");
    int_print_mat(1, n, ipiv, 1);


    blasfeo_dgetr(n, n, &sLU, 0, 0, &sLUt, 0, 0);

    blasfeo_pack_dmat(n, n, I, n, &sI, 0, 0);
    printf("\nI = \n");
    blasfeo_print_dmat(n, n, &sI, 0, 0);

    blasfeo_dtrsm_llnn(n, n, 1.0, &sLUt, 0, 0, &sI, 0, 0, &sH, 0, 0);
    printf("\ninv(U^T) = \n");
    blasfeo_print_dmat(n, n, &sH, 0, 0);

    blasfeo_dtrsm_lunu(n, n, 1.0, &sLUt, 0, 0, &sH, 0, 0, &sH, 0, 0);
    printf("\n(inv(L^T)*inv(U^T)) = \n");
    blasfeo_print_dmat(n, n, &sH, 0, 0);

    blasfeo_drowpei(n, ipiv, &sH);
    printf("\nperm(inv(L^T)*inv(U^T)) = \n");
    blasfeo_print_dmat(n, n, &sH, 0, 0);

    // convert from strmat to column major matrix
    blasfeo_unpack_dmat(n, n, &sH, 0, 0, H, n);

    // print matrix in column-major format
    // H <- G + F*F'
    printf("\ninv(H) = \n");
    d_print_mat(n, n, H, n);

    d_free(F);
    d_free(G);
    d_free(H);
    d_free(I);
    int_free(ipiv);
    v_free_align(memory_strmat);

    /* --------------------------------------- float Lower-Upper Factorization -------------------------------------- */
    n = 16;

    //
    // matrices in column-major format
    //
    float *K; s_zeros(&K, n, n);
    for(ii=0; ii<n*n; ii++)
        K[ii] = ii;
    //s_print_mat(n, n, K, n);

    // spd matrix
    float *M; s_zeros(&M, n, n);
    for(ii=0; ii<n; ii++)
        M[ii*(n+1)] = 1.0;
    //s_print_mat(n, n, M, n);

    // identity
    float *I_s; s_zeros(&I_s, n, n);
    for(ii=0; ii<n; ii++)
        I_s[ii*(n+1)] = 1.0;
    //s_print_mat(n, n, I_s, n);

    // result matrix
    float *N; s_zeros(&N, n, n);
    //s_print_mat(n, n, N, n);

    // permutation indeces
    int *ipiv_s; int_zeros(&ipiv_s, n, 1);

    // work space enough for 5 matrix structs for size n times n
    int size_strmat_s = 5 * blasfeo_memsize_smat(n, n);
    void *memory_strmat_s; v_zeros_align(&memory_strmat_s, size_strmat_s);
    char *ptr_memory_strmat_s = (char *) memory_strmat_s;

    struct blasfeo_smat sK;
    //blasfeo_allocate_smat(n, n, &sA);
    blasfeo_create_smat(n, n, &sK, ptr_memory_strmat_s);
    ptr_memory_strmat_s += sK.memsize;
    // convert from column major matrix to strmat
    blasfeo_pack_smat(n, n, K, n, &sK, 0, 0);
    printf("\nK = \n");
    blasfeo_print_smat(n, n, &sK, 0, 0);

    struct blasfeo_smat sM;
    //blasfeo_allocate_smat(n, n, &sM);
    blasfeo_create_smat(n, n, &sM, ptr_memory_strmat_s);
    ptr_memory_strmat_s += sL.memsize;
    // convert from column major matrix to strmat
    blasfeo_pack_smat(n, n, M, n, &sM, 0, 0);
    printf("\nM = \n");
    blasfeo_print_smat(n, n, &sM, 0, 0);

    struct blasfeo_smat sI_s;
    //blasfeo_allocate_smat(n, n, &sI_s);
    blasfeo_create_smat(n, n, &sI_s, ptr_memory_strmat);
    ptr_memory_strmat_s += sI_s.memsize;
    // convert from column major matrix to strmat

    struct blasfeo_smat sN;
    //blasfeo_allocate_smat(n, n, &sN);
    blasfeo_create_smat(n, n, &sN, ptr_memory_strmat_s);
    ptr_memory_strmat_s += sN.memsize;

    struct blasfeo_smat sLU_s;
    //blasfeo_allocate_smat(n, n, &sLU_s);
    blasfeo_create_smat(n, n, &sLU_s, ptr_memory_strmat_s);
    ptr_memory_strmat_s += sLU_s.memsize;

    blasfeo_sgemm_nt(n, n, n, 1.0, &sK, 0, 0, &sK, 0, 0, 1.0, &sM, 0, 0, &sN, 0, 0);
    printf("\nM+K*K' = \n");
    blasfeo_print_smat(n, n, &sN, 0, 0);

    //	blasfeo_sgetrf_nopivot(n, n, &sD, 0, 0, &sD, 0, 0);
    blasfeo_sgetrf_rp(n, n, &sN, 0, 0, &sLU_s, 0, 0, ipiv_s);
    printf("\nLU = \n");
    blasfeo_print_smat(n, n, &sLU_s, 0, 0);
    printf("\nipiv = \n");
    int_print_mat(1, n, ipiv_s, 1);

    blasfeo_pack_tran_smat(n, n, I_s, n, &sI_s, 0, 0);
    printf("\nI' = \n");
    blasfeo_print_smat(n, n, &sI_s, 0, 0);

    blasfeo_scolpe(n, ipiv_s, &sM);
    printf("\nperm(I') = \n");
    blasfeo_print_smat(n, n, &sM, 0, 0);

    blasfeo_strsm_rltu(n, n, 1.0, &sLU_s, 0, 0, &sM, 0, 0, &sN, 0, 0);
    printf("\nperm(inv(L')) = \n");
    blasfeo_print_smat(n, n, &sN, 0, 0);
    blasfeo_strsm_rutn(n, n, 1.0, &sLU_s, 0, 0, &sN, 0, 0, &sN, 0, 0);
    printf("\ninv(N') = \n");
    blasfeo_print_smat(n, n, &sN, 0, 0);

    // convert from strmat to column major matrix
    blasfeo_unpack_tran_smat(n, n, &sN, 0, 0, N, n);

    // print matrix in column-major format
    printf("\ninv(N) = \n");
    s_print_mat(n, n, N, n);

    /* free memeory */
    s_free(K);
    s_free(M);
    s_free(N);
    s_free(I_s);
    v_free_align(memory_strmat_s);

    /* -------------------------------------------------------------------------------------------------------------- */
    return 0;
}

