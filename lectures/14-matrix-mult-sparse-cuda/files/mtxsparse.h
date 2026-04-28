#ifdef __cplusplus

extern "C" {

#endif

#ifndef MTX_SPARSE
#define MTX_SPARSE

#include <stdio.h>

// COOrdinates
struct mtxCOO { 
    int *row;
    int *col;
    float *data;
    int numRows;
    int numCols;
    int numNonzero;
};

// Compressed Sparse Row
struct mtxCSR {   
    int *rowPtr;
    int *col;
    float *data;
    int numRows;
    int numCols;
    int numNonzero;
};

// ELLiptic (developed by authors of elliptic package)
struct mtxELL {     
    int *col;
    float *data;
    int numRows;
    int numCols;
    int numNonzero;
    int numElements;
    int numElementsInRow;    
};

int mtx_COO_create_from_file(struct mtxCOO *mCOO, FILE *f);
int mtx_COO_free(struct mtxCOO *mCOO);

int mtx_CSR_create_from_mtx_COO(struct mtxCSR *mCSR, struct mtxCOO *mCOO);
int mtx_CSR_free(struct mtxCSR *mCSR);

int mtx_ELL_create_from_mtx_CSR(struct mtxELL *mELL, struct mtxCSR *mCSR);
int mtx_ELL_free(struct mtxELL *mELL);

#endif

#ifdef __cplusplus

}

#endif
