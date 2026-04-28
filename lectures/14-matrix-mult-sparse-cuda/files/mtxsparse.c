#include <stdio.h>
#include <stdlib.h>
#include "mtxsparse.h"

struct mtxMM {
    int row;
    int col;
    float data;
};

int mtx_COO_compare(const void * a, const void * b) {   

    struct mtxMM aa = *(struct mtxMM *)a;
    struct mtxMM bb = *(struct mtxMM *)b;

    if (aa.row < bb.row)
        return -1;
    else if (aa.row > bb.row)
        return +1;
    else if (aa.col < bb.col)
        return -1;
    else if (aa.col > bb.col)
        return +1;
    else 
        return 0;
}

int mtx_COO_create_from_file(struct mtxCOO *mCOO, FILE *f) {

    char line[1024];

    // skip comments
    do 
    {
        if (fgets(line, 1024, f) == NULL) 
            return 1;
    } 
    while (line[0] == '%');
    // get matrix size
    if (sscanf(line, "%d %d %d", &(mCOO->numRows), &(mCOO->numCols), &(mCOO->numNonzero)) != 3)
        return 1;
    // allocate matrix
    struct mtxMM *mMM = (struct mtxMM *)malloc(mCOO->numNonzero * sizeof(struct mtxMM));
    mCOO->data = (float *) malloc(mCOO->numNonzero * sizeof(float));
    mCOO->col = (int *) malloc(mCOO->numNonzero * sizeof(int));
    mCOO->row = (int *) malloc(mCOO->numNonzero * sizeof(int));
    // read data
    for (int i = 0; i < mCOO->numNonzero; i++)
    {
        fscanf(f, "%d %d %f\n", &mMM[i].row, &mMM[i].col, &mMM[i].data);
        mMM[i].row--;  /* adjust from 1-based to 0-based row/column */
        mMM[i].col--;
    }    
    fclose(f);

    // sort elements
    qsort(mMM, mCOO->numNonzero, sizeof(struct mtxMM), mtx_COO_compare);

    // copy to mtx_COO structures (GPU friendly)
    for (int i = 0; i < mCOO->numNonzero; i++)
    {
        mCOO->data[i] = mMM[i].data;
        mCOO->row[i] = mMM[i].row;
        mCOO->col[i] = mMM[i].col;
    }

    free(mMM);

    return 0;
}

int mtx_COO_free(struct mtxCOO *mCOO) {

    free(mCOO->data);
    free(mCOO->col);
    free(mCOO->row);

    return 0;
}

int mtx_CSR_create_from_mtx_COO(struct mtxCSR *mCSR, struct mtxCOO *mCOO) {

    mCSR->numNonzero = mCOO->numNonzero;
    mCSR->numRows = mCOO->numRows;
    mCSR->numCols = mCOO->numCols;

    mCSR->data =  (float *)malloc(mCSR->numNonzero * sizeof(float));
    mCSR->col = (int *)malloc(mCSR->numNonzero * sizeof(int));
    mCSR->rowPtr = (int *)calloc(mCSR->numRows + 1, sizeof(int));
    mCSR->data[0] = mCOO->data[0];
    mCSR->col[0] = mCOO->col[0];
    mCSR->rowPtr[0] = 0;
    mCSR->rowPtr[mCSR->numRows] = mCSR->numNonzero;
    for (int i = 1; i < mCSR->numNonzero; i++)
    {
        mCSR->data[i] = mCOO->data[i];
        mCSR->col[i] = mCOO->col[i];
        if (mCOO->row[i] > mCOO->row[i-1])
        {
            int r = mCOO->row[i];
            while (r > 0 && mCSR->rowPtr[r] == 0)
                mCSR->rowPtr[r--] = i;
        }
    }

    return 0;
}

int mtx_CSR_free(struct mtxCSR *mCSR) {

    free(mCSR->data);
    free(mCSR->col);
    free(mCSR->rowPtr);

    return 0;
}

int mtx_ELL_create_from_mtx_CSR(struct mtxELL *mELL, struct mtxCSR *mCSR) {

    mELL->numNonzero = mCSR->numNonzero;
    mELL->numRows = mCSR->numRows;
    mELL->numCols = mCSR->numCols;
    mELL->numElementsInRow = 0;

    for (int i = 0; i < mELL->numRows; i++)
        if (mELL->numElementsInRow < mCSR->rowPtr[i+1]-mCSR->rowPtr[i]) 
            mELL->numElementsInRow = mCSR->rowPtr[i+1]-mCSR->rowPtr[i];
    mELL->numElements = mELL->numRows * mELL->numElementsInRow;
    mELL->data = (float *)calloc(mELL->numElements, sizeof(float));
    mELL->col = (int *) calloc(mELL->numElements, sizeof(int));    
    for (int i = 0; i < mELL->numRows; i++)
    {
        for (int j = mCSR->rowPtr[i]; j < mCSR->rowPtr[i+1]; j++)
        {            
            int ELL_j = (j - mCSR->rowPtr[i]) * mELL->numRows + i;
            mELL->data[ELL_j] = mCSR->data[j];
            mELL->col[ELL_j] = mCSR->col[j];
        }
    }

    return 0;
}

int mtx_ELL_free(struct mtxELL *mELL) {

    free(mELL->col);
    free(mELL->data);

    return 0;
}
