-- |
-- Module      : BHC.Numeric.Sparse
-- Description : Sparse tensor operations with format-indexed types
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : experimental
--
-- Type-safe sparse tensor operations for efficient memory usage in
-- sparse data scenarios (GNNs, NLP, scientific computing).
--
-- = Overview
--
-- Sparse tensors store only non-zero elements, achieving significant
-- memory and compute savings when data is sparse (>90% zeros).
--
-- = Supported Formats
--
-- * 'COO' - Coordinate format: stores (row, col, value) triples
-- * 'CSR' - Compressed Sparse Row: efficient row-wise access
-- * 'CSC' - Compressed Sparse Column: efficient column-wise access
-- * 'BSR' - Block Sparse Row: for matrices with dense blocks
--
-- = Usage
--
-- @
-- -- Create sparse matrix from COO format
-- let coords = fromList [[0,0], [1,2], [2,1]]
--     values = fromList [1.0, 2.0, 3.0]
--     sparse = mkCOO [3, 3] coords values
--
-- -- Convert to CSR for efficient operations
-- let csr = toCSR sparse
--
-- -- Sparse-dense matrix multiply
-- result = spmm csr denseMatrix
-- @
--
-- = Performance
--
-- * SpMV (sparse matrix-vector): O(nnz) time, O(nnz) space
-- * SpMM (sparse-dense matmul): O(nnz * n) time for (m x k) @ (k x n)
-- * Format conversion: O(nnz log nnz) for COO to CSR

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE StandaloneDeriving #-}

module BHC.Numeric.Sparse (
    -- * Sparse Formats
    SparseFormat(..),

    -- * Sparse Matrix Types
    SparseCOO(..),
    SparseCSR(..),
    SparseCSC(..),
    SparseBSR(..),

    -- * Construction
    mkCOO,
    mkCSR,
    mkCSC,
    fromDense,
    fromTriples,

    -- * Conversion
    toCSR,
    toCSC,
    toCOO,
    toDense,

    -- * Queries
    nnz,
    shape,
    density,
    getRow,
    getCol,

    -- * Sparse Operations
    transpose,
    spmv,
    spmm,
    elemMul,
    add,

    -- * Element Access
    (!!),
    get,
    slice,

    -- * Utilities
    validate,
    compact,
) where

import Prelude hiding ((!!), transpose)
import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector.Unboxed.Mutable as VUM
import qualified Data.Vector as V
import Data.List (sortBy, groupBy)
import Data.Ord (comparing)
import Data.Function (on)
import Control.Monad (forM_, when)
import System.IO.Unsafe (unsafePerformIO)

import BHC.Numeric.Tensor (Tensor)
import qualified BHC.Numeric.Tensor as T

-- ============================================================================
-- Sparse Format Types
-- ============================================================================

-- | Sparse storage format at type level.
data SparseFormat
    = COO  -- ^ Coordinate format: (row, col, value) triples
    | CSR  -- ^ Compressed Sparse Row
    | CSC  -- ^ Compressed Sparse Column
    | BSR  -- ^ Block Sparse Row
    deriving (Show, Eq)

-- ============================================================================
-- Sparse Matrix Data Types
-- ============================================================================

-- | Coordinate (COO) sparse matrix format.
--
-- Stores explicit (row, col, value) triples for each non-zero.
-- Good for construction and random updates, but slower for linear algebra.
--
-- Invariants:
-- - rowIndices, colIndices, values all have the same length (nnz)
-- - 0 <= rowIndices[i] < rows
-- - 0 <= colIndices[i] < cols
data SparseCOO a = SparseCOO
    { cooRows       :: !Int             -- ^ Number of rows
    , cooCols       :: !Int             -- ^ Number of columns
    , cooRowIndices :: !(VU.Vector Int) -- ^ Row indices of non-zeros
    , cooColIndices :: !(VU.Vector Int) -- ^ Column indices of non-zeros
    , cooValues     :: !(VU.Vector a)   -- ^ Non-zero values
    } deriving (Show, Eq)

-- | Compressed Sparse Row (CSR) format.
--
-- The standard format for sparse matrix operations. Stores:
-- - rowPtrs: indices into colIndices/values for each row
-- - colIndices: column indices of non-zeros
-- - values: non-zero values
--
-- Invariants:
-- - rowPtrs has length (rows + 1)
-- - rowPtrs[0] = 0
-- - rowPtrs[rows] = nnz
-- - rowPtrs is monotonically non-decreasing
data SparseCSR a = SparseCSR
    { csrRows       :: !Int             -- ^ Number of rows
    , csrCols       :: !Int             -- ^ Number of columns
    , csrRowPtrs    :: !(VU.Vector Int) -- ^ Row pointers (length = rows + 1)
    , csrColIndices :: !(VU.Vector Int) -- ^ Column indices of non-zeros
    , csrValues     :: !(VU.Vector a)   -- ^ Non-zero values
    } deriving (Show, Eq)

-- | Compressed Sparse Column (CSC) format.
--
-- Transpose of CSR - efficient for column-wise access.
data SparseCSC a = SparseCSC
    { cscRows       :: !Int             -- ^ Number of rows
    , cscCols       :: !Int             -- ^ Number of columns
    , cscColPtrs    :: !(VU.Vector Int) -- ^ Column pointers (length = cols + 1)
    , cscRowIndices :: !(VU.Vector Int) -- ^ Row indices of non-zeros
    , cscValues     :: !(VU.Vector a)   -- ^ Non-zero values
    } deriving (Show, Eq)

-- | Block Sparse Row (BSR) format.
--
-- For matrices with dense blocks. Useful for finite element methods
-- and some neural network architectures.
data SparseBSR a = SparseBSR
    { bsrRows       :: !Int             -- ^ Number of block rows
    , bsrCols       :: !Int             -- ^ Number of block columns
    , bsrBlockRows  :: !Int             -- ^ Block height
    , bsrBlockCols  :: !Int             -- ^ Block width
    , bsrRowPtrs    :: !(VU.Vector Int) -- ^ Block row pointers
    , bsrColIndices :: !(VU.Vector Int) -- ^ Block column indices
    , bsrValues     :: !(V.Vector (VU.Vector a)) -- ^ Dense blocks
    } deriving (Show, Eq)

-- ============================================================================
-- Construction
-- ============================================================================

-- | Create a COO sparse matrix from coordinate arrays.
--
-- >>> mkCOO (3, 3) (fromList [0, 1, 2]) (fromList [0, 1, 2]) (fromList [1.0, 2.0, 3.0])
-- SparseCOO {cooRows = 3, cooCols = 3, ...}
mkCOO :: VU.Unbox a
      => (Int, Int)        -- ^ (rows, cols)
      -> VU.Vector Int     -- ^ Row indices
      -> VU.Vector Int     -- ^ Column indices
      -> VU.Vector a       -- ^ Values
      -> Either String (SparseCOO a)
mkCOO (rows, cols) rowIdx colIdx vals
    | VU.length rowIdx /= VU.length colIdx =
        Left "Row and column indices must have same length"
    | VU.length rowIdx /= VU.length vals =
        Left "Indices and values must have same length"
    | VU.any (< 0) rowIdx || VU.any (>= rows) rowIdx =
        Left "Row indices out of bounds"
    | VU.any (< 0) colIdx || VU.any (>= cols) colIdx =
        Left "Column indices out of bounds"
    | otherwise = Right $ SparseCOO rows cols rowIdx colIdx vals

-- | Create a CSR sparse matrix directly.
--
-- This is the most efficient constructor if you already have CSR data.
mkCSR :: VU.Unbox a
      => (Int, Int)        -- ^ (rows, cols)
      -> VU.Vector Int     -- ^ Row pointers
      -> VU.Vector Int     -- ^ Column indices
      -> VU.Vector a       -- ^ Values
      -> Either String (SparseCSR a)
mkCSR (rows, cols) rowPtrs colIdx vals
    | VU.length rowPtrs /= rows + 1 =
        Left "Row pointers must have length (rows + 1)"
    | VU.head rowPtrs /= 0 =
        Left "First row pointer must be 0"
    | VU.length colIdx /= VU.length vals =
        Left "Column indices and values must have same length"
    | VU.last rowPtrs /= VU.length vals =
        Left "Last row pointer must equal nnz"
    | otherwise = Right $ SparseCSR rows cols rowPtrs colIdx vals

-- | Create a CSC sparse matrix directly.
mkCSC :: VU.Unbox a
      => (Int, Int)        -- ^ (rows, cols)
      -> VU.Vector Int     -- ^ Column pointers
      -> VU.Vector Int     -- ^ Row indices
      -> VU.Vector a       -- ^ Values
      -> Either String (SparseCSC a)
mkCSC (rows, cols) colPtrs rowIdx vals
    | VU.length colPtrs /= cols + 1 =
        Left "Column pointers must have length (cols + 1)"
    | VU.head colPtrs /= 0 =
        Left "First column pointer must be 0"
    | VU.length rowIdx /= VU.length vals =
        Left "Row indices and values must have same length"
    | VU.last colPtrs /= VU.length vals =
        Left "Last column pointer must equal nnz"
    | otherwise = Right $ SparseCSC rows cols colPtrs rowIdx vals

-- | Create sparse matrix from (row, col, value) triples.
--
-- >>> fromTriples (3, 3) [(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)]
-- Right (SparseCOO ...)
fromTriples :: VU.Unbox a
            => (Int, Int)         -- ^ (rows, cols)
            -> [(Int, Int, a)]    -- ^ (row, col, value) triples
            -> Either String (SparseCOO a)
fromTriples (rows, cols) triples =
    let (rs, cs, vs) = unzip3 triples
    in mkCOO (rows, cols)
             (VU.fromList rs)
             (VU.fromList cs)
             (VU.fromList vs)

-- | Convert dense tensor to sparse COO format.
--
-- Only stores non-zero elements.
fromDense :: (VU.Unbox a, Eq a, Num a)
          => Tensor a
          -> SparseCOO a
fromDense tensor =
    let sh = T.shape tensor
        (rows, cols) = case sh of
            [r, c] -> (r, c)
            [r]    -> (r, 1)
            _      -> error "fromDense: expected 1D or 2D tensor"
        -- Collect non-zero elements
        triples = [ (i, j, v)
                  | i <- [0..rows-1]
                  , j <- [0..cols-1]
                  , let v = T.index tensor [i, j]
                  , v /= 0
                  ]
        (rs, cs, vs) = unzip3 triples
    in SparseCOO rows cols
                 (VU.fromList rs)
                 (VU.fromList cs)
                 (VU.fromList vs)

-- ============================================================================
-- Format Conversion
-- ============================================================================

-- | Convert COO to CSR format.
--
-- Time: O(nnz log nnz) for sorting
-- Space: O(nnz)
toCSR :: VU.Unbox a => SparseCOO a -> SparseCSR a
toCSR coo = unsafePerformIO $ do
    let n = VU.length (cooValues coo)
        rows = cooRows coo
        cols = cooCols coo

    -- Sort by (row, col)
    let indices = [0..n-1]
        sorted = sortBy (comparing $ \i ->
            (cooRowIndices coo VU.! i, cooColIndices coo VU.! i)) indices

    -- Build CSR arrays
    rowPtrsMut <- VUM.replicate (rows + 1) 0
    colIndicesMut <- VUM.new n
    valuesMut <- VUM.new n

    -- Count elements per row
    forM_ sorted $ \srcIdx -> do
        let row = cooRowIndices coo VU.! srcIdx
        VUM.modify rowPtrsMut (+1) (row + 1)

    -- Cumulative sum for row pointers
    forM_ [1..rows] $ \i -> do
        prev <- VUM.read rowPtrsMut (i - 1)
        curr <- VUM.read rowPtrsMut i
        VUM.write rowPtrsMut i (prev + curr)

    -- Reset for filling
    currentPos <- VUM.replicate rows 0
    forM_ [0..rows-1] $ \row -> do
        ptr <- VUM.read rowPtrsMut row
        VUM.write currentPos row ptr

    -- Fill colIndices and values
    forM_ sorted $ \srcIdx -> do
        let row = cooRowIndices coo VU.! srcIdx
            col = cooColIndices coo VU.! srcIdx
            val = cooValues coo VU.! srcIdx
        pos <- VUM.read currentPos row
        VUM.write colIndicesMut pos col
        VUM.write valuesMut pos val
        VUM.modify currentPos (+1) row

    rowPtrs <- VU.freeze rowPtrsMut
    colIndices <- VU.freeze colIndicesMut
    values <- VU.freeze valuesMut

    return $ SparseCSR rows cols rowPtrs colIndices values

-- | Convert COO to CSC format.
--
-- Time: O(nnz log nnz) for sorting
-- Space: O(nnz)
toCSC :: VU.Unbox a => SparseCOO a -> SparseCSC a
toCSC coo = unsafePerformIO $ do
    let n = VU.length (cooValues coo)
        rows = cooRows coo
        cols = cooCols coo

    -- Sort by (col, row)
    let indices = [0..n-1]
        sorted = sortBy (comparing $ \i ->
            (cooColIndices coo VU.! i, cooRowIndices coo VU.! i)) indices

    -- Build CSC arrays
    colPtrsMut <- VUM.replicate (cols + 1) 0
    rowIndicesMut <- VUM.new n
    valuesMut <- VUM.new n

    -- Count elements per column
    forM_ sorted $ \srcIdx -> do
        let col = cooColIndices coo VU.! srcIdx
        VUM.modify colPtrsMut (+1) (col + 1)

    -- Cumulative sum for column pointers
    forM_ [1..cols] $ \i -> do
        prev <- VUM.read colPtrsMut (i - 1)
        curr <- VUM.read colPtrsMut i
        VUM.write colPtrsMut i (prev + curr)

    -- Reset for filling
    currentPos <- VUM.replicate cols 0
    forM_ [0..cols-1] $ \col -> do
        ptr <- VUM.read colPtrsMut col
        VUM.write currentPos col ptr

    -- Fill rowIndices and values
    forM_ sorted $ \srcIdx -> do
        let row = cooRowIndices coo VU.! srcIdx
            col = cooColIndices coo VU.! srcIdx
            val = cooValues coo VU.! srcIdx
        pos <- VUM.read currentPos col
        VUM.write rowIndicesMut pos row
        VUM.write valuesMut pos val
        VUM.modify currentPos (+1) col

    colPtrs <- VU.freeze colPtrsMut
    rowIndices <- VU.freeze rowIndicesMut
    values <- VU.freeze valuesMut

    return $ SparseCSC rows cols colPtrs rowIndices values

-- | Convert CSR back to COO format.
toCOO :: VU.Unbox a => SparseCSR a -> SparseCOO a
toCOO csr =
    let n = VU.length (csrValues csr)
        rows = csrRows csr
        cols = csrCols csr
        -- Expand row pointers to row indices
        rowIndices = VU.create $ do
            v <- VUM.new n
            forM_ [0..rows-1] $ \row -> do
                let start = csrRowPtrs csr VU.! row
                    end = csrRowPtrs csr VU.! (row + 1)
                forM_ [start..end-1] $ \i ->
                    VUM.write v i row
            return v
    in SparseCOO rows cols rowIndices (csrColIndices csr) (csrValues csr)

-- | Convert sparse matrix to dense tensor.
--
-- Warning: This may use significant memory for large sparse matrices.
toDense :: (VU.Unbox a, Num a) => SparseCSR a -> Tensor a
toDense csr =
    let rows = csrRows csr
        cols = csrCols csr
        -- Initialize dense tensor with zeros
        dense = T.zeros [rows, cols]
        -- Fill in non-zero values
    in unsafePerformIO $ do
        let denseList = concat
              [ [ (i, j, csrValues csr VU.! k)
                | k <- [csrRowPtrs csr VU.! i .. csrRowPtrs csr VU.! (i+1) - 1]
                , let j = csrColIndices csr VU.! k
                ]
              | i <- [0..rows-1]
              ]
        -- Build tensor from list
        return $ foldr (\(i, j, v) t -> T.setIndex t [i, j] v) dense denseList

-- ============================================================================
-- Queries
-- ============================================================================

-- | Number of non-zero elements.
class Sparse s where
    nnz :: s a -> Int
    shape :: s a -> (Int, Int)

instance Sparse SparseCOO where
    nnz coo = VU.length (cooValues coo)
    shape coo = (cooRows coo, cooCols coo)

instance Sparse SparseCSR where
    nnz csr = VU.length (csrValues csr)
    shape csr = (csrRows csr, csrCols csr)

instance Sparse SparseCSC where
    nnz csc = VU.length (cscValues csc)
    shape csc = (cscRows csc, cscCols csc)

-- | Density of the sparse matrix (nnz / total elements).
density :: Sparse s => s a -> Double
density s =
    let (rows, cols) = shape s
        total = rows * cols
    in if total == 0
       then 0.0
       else fromIntegral (nnz s) / fromIntegral total

-- | Get a specific row from CSR matrix as a sparse vector.
getRow :: VU.Unbox a => SparseCSR a -> Int -> (VU.Vector Int, VU.Vector a)
getRow csr row
    | row < 0 || row >= csrRows csr = error "getRow: index out of bounds"
    | otherwise =
        let start = csrRowPtrs csr VU.! row
            end = csrRowPtrs csr VU.! (row + 1)
        in ( VU.slice start (end - start) (csrColIndices csr)
           , VU.slice start (end - start) (csrValues csr)
           )

-- | Get a specific column from CSC matrix as a sparse vector.
getCol :: VU.Unbox a => SparseCSC a -> Int -> (VU.Vector Int, VU.Vector a)
getCol csc col
    | col < 0 || col >= cscCols csc = error "getCol: index out of bounds"
    | otherwise =
        let start = cscColPtrs csc VU.! col
            end = cscColPtrs csc VU.! (col + 1)
        in ( VU.slice start (end - start) (cscRowIndices csc)
           , VU.slice start (end - start) (cscValues csc)
           )

-- ============================================================================
-- Element Access
-- ============================================================================

-- | Index a sparse matrix element.
--
-- Returns 0 if the element is not stored (assumed sparse).
-- Time: O(log nnz) for CSR/CSC, O(nnz) for COO.
(!!) :: (VU.Unbox a, Num a) => SparseCSR a -> (Int, Int) -> a
csr !! (row, col)
    | row < 0 || row >= csrRows csr = error "!!: row index out of bounds"
    | col < 0 || col >= csrCols csr = error "!!: column index out of bounds"
    | otherwise =
        let start = csrRowPtrs csr VU.! row
            end = csrRowPtrs csr VU.! (row + 1)
            -- Binary search for column in this row
            colSlice = VU.slice start (end - start) (csrColIndices csr)
            valSlice = VU.slice start (end - start) (csrValues csr)
        in case binarySearch colSlice col of
             Just idx -> valSlice VU.! idx
             Nothing  -> 0

-- | Safe element access with Maybe result.
get :: (VU.Unbox a, Num a) => SparseCSR a -> (Int, Int) -> Maybe a
get csr (row, col)
    | row < 0 || row >= csrRows csr = Nothing
    | col < 0 || col >= csrCols csr = Nothing
    | otherwise =
        let start = csrRowPtrs csr VU.! row
            end = csrRowPtrs csr VU.! (row + 1)
            colSlice = VU.slice start (end - start) (csrColIndices csr)
            valSlice = VU.slice start (end - start) (csrValues csr)
        in case binarySearch colSlice col of
             Just idx -> Just (valSlice VU.! idx)
             Nothing  -> Just 0

-- | Extract a submatrix (slice) from a sparse matrix.
slice :: VU.Unbox a
      => SparseCSR a
      -> (Int, Int)  -- ^ (startRow, startCol)
      -> (Int, Int)  -- ^ (endRow, endCol) exclusive
      -> SparseCSR a
slice csr (startRow, startCol) (endRow, endCol) = unsafePerformIO $ do
    let newRows = endRow - startRow
        newCols = endCol - startCol

    -- Collect elements in range
    let triples =
          [ (i - startRow, j - startCol, csrValues csr VU.! k)
          | i <- [startRow..endRow-1]
          , let rowStart = csrRowPtrs csr VU.! i
                rowEnd = csrRowPtrs csr VU.! (i + 1)
          , k <- [rowStart..rowEnd-1]
          , let j = csrColIndices csr VU.! k
          , j >= startCol && j < endCol
          ]

    -- Build new CSR
    let coo = case fromTriples (newRows, newCols) triples of
                Right c -> c
                Left _  -> error "slice: internal error"
    return $ toCSR coo

-- ============================================================================
-- Sparse Operations
-- ============================================================================

-- | Transpose a sparse matrix.
--
-- For CSR, this produces CSC (and vice versa), then converts back.
-- Time: O(nnz)
transpose :: VU.Unbox a => SparseCSR a -> SparseCSR a
transpose csr =
    -- CSR transpose is equivalent to CSC with rows/cols swapped
    let csc = SparseCSC (csrCols csr) (csrRows csr)
                        (csrRowPtrs csr) (csrColIndices csr) (csrValues csr)
        -- Convert transposed CSC back to CSR
        coo = SparseCOO (cscCols csc) (cscRows csc)
                        (cscRowIndices csc)
                        (VU.create $ do
                           v <- VUM.new (VU.length $ cscValues csc)
                           forM_ [0..cscCols csc - 1] $ \col -> do
                             let start = cscColPtrs csc VU.! col
                                 end = cscColPtrs csc VU.! (col + 1)
                             forM_ [start..end-1] $ \i ->
                               VUM.write v i col
                           return v)
                        (cscValues csc)
    in toCSR coo

-- | Sparse matrix-vector multiplication (SpMV).
--
-- y = A * x where A is sparse CSR and x is dense.
--
-- Time: O(nnz)
-- Space: O(rows) for output
spmv :: (VU.Unbox a, Num a) => SparseCSR a -> VU.Vector a -> VU.Vector a
spmv csr x
    | csrCols csr /= VU.length x =
        error "spmv: dimension mismatch"
    | otherwise = VU.create $ do
        y <- VUM.replicate (csrRows csr) 0

        forM_ [0..csrRows csr - 1] $ \row -> do
            let start = csrRowPtrs csr VU.! row
                end = csrRowPtrs csr VU.! (row + 1)
            sum <- foldM (\acc k -> do
                let col = csrColIndices csr VU.! k
                    val = csrValues csr VU.! k
                    xVal = x VU.! col
                return $! acc + val * xVal
              ) 0 [start..end-1]
            VUM.write y row sum

        return y
  where
    foldM f z [] = return z
    foldM f z (x:xs) = do
        z' <- f z x
        foldM f z' xs

-- | Sparse-dense matrix multiplication (SpMM).
--
-- C = A * B where A is sparse CSR and B is dense.
--
-- Time: O(nnz * n) for A(m x k) and B(k x n)
-- Space: O(m * n) for output
spmm :: (VU.Unbox a, Num a) => SparseCSR a -> Tensor a -> Tensor a
spmm csr b =
    let [k, n] = T.shape b
    in if csrCols csr /= k
       then error "spmm: dimension mismatch"
       else T.fromListFlat [csrRows csr, n] $ concat
         [ [ sum [ csrValues csr VU.! idx * T.index b [csrColIndices csr VU.! idx, j]
                 | idx <- [csrRowPtrs csr VU.! i .. csrRowPtrs csr VU.! (i+1) - 1]
                 ]
           | j <- [0..n-1]
           ]
         | i <- [0..csrRows csr - 1]
         ]

-- | Element-wise multiplication of two sparse matrices.
--
-- Only stores non-zeros present in both matrices.
elemMul :: (VU.Unbox a, Num a) => SparseCSR a -> SparseCSR a -> SparseCSR a
elemMul a b
    | shape a /= shape b = error "elemMul: shape mismatch"
    | otherwise =
        let (rows, cols) = shape a
            triples =
              [ (i, j, va * vb)
              | i <- [0..rows-1]
              , let (colsA, valsA) = getRow a i
                    (colsB, valsB) = getRow b i
              , (j, va) <- zip (VU.toList colsA) (VU.toList valsA)
              , (j', vb) <- zip (VU.toList colsB) (VU.toList valsB)
              , j == j'
              ]
            coo = case fromTriples (rows, cols) triples of
                    Right c -> c
                    Left _ -> error "elemMul: internal error"
        in toCSR coo

-- | Add two sparse matrices.
--
-- The result may have more non-zeros than either input.
add :: (VU.Unbox a, Num a, Eq a) => SparseCSR a -> SparseCSR a -> SparseCSR a
add a b
    | shape a /= shape b = error "add: shape mismatch"
    | otherwise =
        let (rows, cols) = shape a
            -- Merge rows from both matrices
            triples = concat
              [ mergeRow i (getRow a i) (getRow b i)
              | i <- [0..rows-1]
              ]
            coo = case fromTriples (rows, cols) triples of
                    Right c -> c
                    Left _ -> error "add: internal error"
        in toCSR coo
  where
    mergeRow row (colsA, valsA) (colsB, valsB) =
        let combined = mergeVecs (zip (VU.toList colsA) (VU.toList valsA))
                                 (zip (VU.toList colsB) (VU.toList valsB))
        in [ (row, col, val) | (col, val) <- combined, val /= 0 ]

    mergeVecs [] bs = bs
    mergeVecs as [] = as
    mergeVecs ((ca, va):as) ((cb, vb):bs)
        | ca < cb   = (ca, va) : mergeVecs as ((cb, vb):bs)
        | ca > cb   = (cb, vb) : mergeVecs ((ca, va):as) bs
        | otherwise = (ca, va + vb) : mergeVecs as bs

-- ============================================================================
-- Utilities
-- ============================================================================

-- | Validate sparse matrix invariants.
validate :: VU.Unbox a => SparseCSR a -> Either String ()
validate csr = do
    let rows = csrRows csr
        cols = csrCols csr
        nz = VU.length (csrValues csr)

    -- Check row pointers
    when (VU.length (csrRowPtrs csr) /= rows + 1) $
        Left "Invalid row pointers length"

    when (VU.head (csrRowPtrs csr) /= 0) $
        Left "First row pointer must be 0"

    when (VU.last (csrRowPtrs csr) /= nz) $
        Left "Last row pointer must equal nnz"

    -- Check monotonicity
    forM_ [1..rows] $ \i -> do
        let prev = csrRowPtrs csr VU.! (i - 1)
            curr = csrRowPtrs csr VU.! i
        when (curr < prev) $
            Left "Row pointers must be non-decreasing"

    -- Check column indices in bounds
    forM_ [0..nz-1] $ \i -> do
        let col = csrColIndices csr VU.! i
        when (col < 0 || col >= cols) $
            Left "Column index out of bounds"

    return ()

-- | Remove duplicate entries and sum their values.
--
-- This is useful after incremental construction.
compact :: (VU.Unbox a, Num a, Eq a) => SparseCOO a -> SparseCOO a
compact coo =
    let n = VU.length (cooValues coo)
        -- Group by (row, col)
        indexed = zip3 (VU.toList $ cooRowIndices coo)
                       (VU.toList $ cooColIndices coo)
                       (VU.toList $ cooValues coo)
        sorted = sortBy (comparing $ \(r, c, _) -> (r, c)) indexed
        grouped = groupBy ((==) `on` \(r, c, _) -> (r, c)) sorted
        -- Sum duplicates
        merged = [ (r, c, sum vs)
                 | grp <- grouped
                 , let (r, c, _) = head grp
                       vs = [v | (_, _, v) <- grp]
                 ]
        -- Filter zeros
        nonZero = [(r, c, v) | (r, c, v) <- merged, v /= 0]
        (rs, cs, vs) = unzip3 nonZero
    in SparseCOO (cooRows coo) (cooCols coo)
                 (VU.fromList rs)
                 (VU.fromList cs)
                 (VU.fromList vs)

-- ============================================================================
-- Internal Helpers
-- ============================================================================

-- | Binary search in a sorted vector.
binarySearch :: VU.Vector Int -> Int -> Maybe Int
binarySearch vec target = go 0 (VU.length vec)
  where
    go lo hi
        | lo >= hi = Nothing
        | otherwise =
            let mid = lo + (hi - lo) `div` 2
                val = vec VU.! mid
            in if val == target
               then Just mid
               else if val < target
                    then go (mid + 1) hi
                    else go lo mid
