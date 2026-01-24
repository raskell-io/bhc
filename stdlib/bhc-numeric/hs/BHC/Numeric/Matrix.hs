-- |
-- Module      : BHC.Numeric.Matrix
-- Description : Dense matrix operations
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : stable
--
-- Dense 2D matrices with BLAS-accelerated operations.
-- Matrices are stored in row-major order.

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module BHC.Numeric.Matrix (
    -- * Matrix type
    Matrix,

    -- * Construction
    zeros, ones, full,
    fromLists, fromList,
    fromRows, fromCols,
    identity, eye,
    diag, diagFrom,

    -- * Properties
    rows, cols, size,
    shape,

    -- * Indexing
    (!), (!?),
    row, col,
    getRow, getCol,
    getDiag,

    -- * Slicing
    submatrix,
    takeRows, dropRows,
    takeCols, dropCols,

    -- * Combining
    (|||), (---),
    hcat, vcat,
    fromBlocks,

    -- * Element-wise operations
    map, imap,
    zipWith, izipWith,

    -- * Matrix operations
    transpose, (^.),
    (+.), (-.), (*.), (/.),
    scale,

    -- * Matrix multiplication
    (@@), mul, mulV,

    -- * Linear algebra
    trace, det, rank,
    inv, pinv,
    solve, lstsq,

    -- * Decompositions
    lu, qr, svd,
    cholesky,
    eig, eigvals,

    -- * Norms
    norm1, norm2, normInf, normFrob,

    -- * Folds
    sum, product,
    maximum, minimum,
    sumRows, sumCols,
    meanRows, meanCols,

    -- * Conversion
    toLists, toList,
    flatten,
    asVector, asColumn,

    -- * Type class
    MatrixElem,
) where

import BHC.Prelude hiding (
    map, zipWith, sum, product, maximum, minimum
    )
import qualified BHC.Prelude as P
import qualified BHC.Numeric.Vector as V
import Foreign.Ptr (Ptr, nullPtr, FunPtr)
import Foreign.ForeignPtr (ForeignPtr, newForeignPtr, withForeignPtr)
import Foreign.Marshal.Array (withArrayLen)
import System.IO.Unsafe (unsafePerformIO)

-- ============================================================
-- FFI Imports for f64 matrices
-- ============================================================

foreign import ccall unsafe "bhc_matrix_from_f64"
    c_matrix_from_f64 :: Ptr Double -> Int -> Int -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_free_f64"
    c_matrix_free_f64 :: Ptr MatrixData -> IO ()

foreign import ccall unsafe "&bhc_matrix_free_f64"
    c_matrix_finalizer_f64 :: FunPtr (Ptr MatrixData -> IO ())

foreign import ccall unsafe "bhc_matrix_rows_f64"
    c_matrix_rows_f64 :: Ptr MatrixData -> IO Int

foreign import ccall unsafe "bhc_matrix_cols_f64"
    c_matrix_cols_f64 :: Ptr MatrixData -> IO Int

foreign import ccall unsafe "bhc_matrix_get_f64"
    c_matrix_get_f64 :: Ptr MatrixData -> Int -> Int -> IO Double

foreign import ccall unsafe "bhc_matrix_zeros_f64"
    c_matrix_zeros_f64 :: Int -> Int -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_identity_f64"
    c_matrix_identity_f64 :: Int -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_matmul_f64"
    c_matrix_matmul_f64 :: Ptr MatrixData -> Ptr MatrixData -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_transpose_f64"
    c_matrix_transpose_f64 :: Ptr MatrixData -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_add_f64"
    c_matrix_add_f64 :: Ptr MatrixData -> Ptr MatrixData -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_scale_f64"
    c_matrix_scale_f64 :: Ptr MatrixData -> Double -> IO (Ptr MatrixData)

foreign import ccall unsafe "bhc_matrix_trace_f64"
    c_matrix_trace_f64 :: Ptr MatrixData -> IO Double

foreign import ccall unsafe "bhc_matrix_norm_f64"
    c_matrix_norm_f64 :: Ptr MatrixData -> IO Double

-- ============================================================
-- Matrix Type
-- ============================================================

-- | A dense 2D matrix using foreign memory.
data Matrix a = Matrix
    { matPtr     :: !(ForeignPtr MatrixData)
    , matRows    :: !Int
    , matCols    :: !Int
    , matStride  :: !Int  -- Row stride
    , matOffset  :: !Int
    }

-- | Internal matrix storage (opaque Rust type).
data MatrixData

-- | Type class for matrix element operations
class MatrixElem a where
    matrixFromList :: Int -> Int -> [a] -> IO (Matrix a)
    matrixGet :: Matrix a -> Int -> Int -> IO a
    matrixZeros :: Int -> Int -> IO (Matrix a)
    matrixIdentity :: Int -> IO (Matrix a)
    matrixMatmul :: Matrix a -> Matrix a -> IO (Matrix a)
    matrixTranspose :: Matrix a -> IO (Matrix a)
    matrixAdd :: Matrix a -> Matrix a -> IO (Matrix a)
    matrixScale :: a -> Matrix a -> IO (Matrix a)
    matrixTrace :: Matrix a -> IO a
    matrixNorm :: Matrix a -> IO a

instance MatrixElem Double where
    matrixFromList r c xs = do
        withArrayLen xs $ \len ptr -> do
            if len /= r * c
                then error "Matrix dimensions don't match data length"
                else do
                    mptr <- c_matrix_from_f64 ptr r c
                    if mptr == nullPtr
                        then error "Failed to create matrix"
                        else do
                            fp <- newForeignPtr c_matrix_finalizer_f64 mptr
                            return $ Matrix fp r c c 0
    matrixGet (Matrix fp _ _ _ _) row col = withForeignPtr fp $ \ptr ->
        c_matrix_get_f64 ptr row col
    matrixZeros r c = do
        mptr <- c_matrix_zeros_f64 r c
        if mptr == nullPtr
            then error "Failed to create zeros matrix"
            else do
                fp <- newForeignPtr c_matrix_finalizer_f64 mptr
                return $ Matrix fp r c c 0
    matrixIdentity n = do
        mptr <- c_matrix_identity_f64 n
        if mptr == nullPtr
            then error "Failed to create identity matrix"
            else do
                fp <- newForeignPtr c_matrix_finalizer_f64 mptr
                return $ Matrix fp n n n 0
    matrixMatmul (Matrix fp1 r1 c1 _ _) (Matrix fp2 r2 c2 _ _) =
        withForeignPtr fp1 $ \p1 ->
        withForeignPtr fp2 $ \p2 -> do
            mptr <- c_matrix_matmul_f64 p1 p2
            if mptr == nullPtr
                then error "Matrix multiplication failed (dimension mismatch)"
                else do
                    fp <- newForeignPtr c_matrix_finalizer_f64 mptr
                    return $ Matrix fp r1 c2 c2 0
    matrixTranspose (Matrix fp r c _ _) =
        withForeignPtr fp $ \ptr -> do
            mptr <- c_matrix_transpose_f64 ptr
            if mptr == nullPtr
                then error "Matrix transpose failed"
                else do
                    fp' <- newForeignPtr c_matrix_finalizer_f64 mptr
                    return $ Matrix fp' c r r 0
    matrixAdd (Matrix fp1 r c _ _) (Matrix fp2 _ _ _ _) =
        withForeignPtr fp1 $ \p1 ->
        withForeignPtr fp2 $ \p2 -> do
            mptr <- c_matrix_add_f64 p1 p2
            if mptr == nullPtr
                then error "Matrix addition failed (dimension mismatch)"
                else do
                    fp <- newForeignPtr c_matrix_finalizer_f64 mptr
                    return $ Matrix fp r c c 0
    matrixScale s (Matrix fp r c _ _) =
        withForeignPtr fp $ \ptr -> do
            mptr <- c_matrix_scale_f64 ptr s
            if mptr == nullPtr
                then error "Matrix scale failed"
                else do
                    fp' <- newForeignPtr c_matrix_finalizer_f64 mptr
                    return $ Matrix fp' r c c 0
    matrixTrace (Matrix fp _ _ _ _) = withForeignPtr fp c_matrix_trace_f64
    matrixNorm (Matrix fp _ _ _ _) = withForeignPtr fp c_matrix_norm_f64

-- ============================================================
-- Construction
-- ============================================================

-- | Create matrix of zeros.
zeros :: MatrixElem a => Int -> Int -> Matrix a
zeros r c = unsafePerformIO $ matrixZeros r c
{-# NOINLINE zeros #-}

-- | Create matrix of ones.
ones :: (Num a, MatrixElem a) => Int -> Int -> Matrix a
ones r c = full r c 1

-- | Create matrix filled with value.
full :: MatrixElem a => Int -> Int -> a -> Matrix a
full r c x = fromList r c (P.replicate (r * c) x)

-- | Create matrix from nested lists.
--
-- >>> fromLists [[1, 2], [3, 4]]
-- Matrix 2x2 [[1, 2], [3, 4]]
fromLists :: MatrixElem a => [[a]] -> Matrix a
fromLists xss =
    let r = P.length xss
        c = if r > 0 then P.length (P.head xss) else 0
    in fromList r c (P.concat xss)

-- | Create matrix from flat list with dimensions.
fromList :: MatrixElem a => Int -> Int -> [a] -> Matrix a
fromList r c xs = unsafePerformIO $ matrixFromList r c xs
{-# NOINLINE fromList #-}

-- | Create matrix from row vectors.
fromRows :: (V.VectorElem a, MatrixElem a) => [V.Vector a] -> Matrix a
fromRows vs =
    let r = P.length vs
        c = if r > 0 then V.length (P.head vs) else 0
    in fromList r c (P.concatMap V.toList vs)

-- | Create matrix from column vectors.
fromCols :: (V.VectorElem a, MatrixElem a) => [V.Vector a] -> Matrix a
fromCols vs = transpose (fromRows vs)

-- | Identity matrix.
--
-- >>> identity 3
-- Matrix 3x3 [[1,0,0], [0,1,0], [0,0,1]]
identity :: MatrixElem a => Int -> Matrix a
identity n = unsafePerformIO $ matrixIdentity n
{-# NOINLINE identity #-}

-- | Identity matrix (alias for identity).
eye :: MatrixElem a => Int -> Matrix a
eye = identity

-- | Diagonal matrix from list.
diag :: (Num a, MatrixElem a) => [a] -> Matrix a
diag xs =
    let n = P.length xs
        indices = [(i, j) | i <- [0..n-1], j <- [0..n-1]]
        elems = [if i == j then xs P.!! i else 0 | (i, j) <- indices]
    in fromList n n elems

-- | Diagonal matrix from vector.
diagFrom :: (Num a, V.VectorElem a, MatrixElem a) => V.Vector a -> Matrix a
diagFrom v = diag (V.toList v)

-- ============================================================
-- Properties
-- ============================================================

-- | Number of rows.
rows :: Matrix a -> Int
rows = matRows

-- | Number of columns.
cols :: Matrix a -> Int
cols = matCols

-- | Total number of elements.
size :: Matrix a -> Int
size m = rows m * cols m

-- | Shape as (rows, cols).
shape :: Matrix a -> (Int, Int)
shape m = (rows m, cols m)

-- ============================================================
-- Indexing
-- ============================================================

-- | Index into matrix (unsafe).
(!) :: MatrixElem a => Matrix a -> (Int, Int) -> a
m ! (i, j) = unsafePerformIO $ matrixGet m i j
{-# NOINLINE (!) #-}

-- | Index into matrix (safe).
(!?) :: MatrixElem a => Matrix a -> (Int, Int) -> Maybe a
m !? (i, j)
    | i < 0 || i >= rows m = Nothing
    | j < 0 || j >= cols m = Nothing
    | otherwise = Just (m ! (i, j))

-- | Extract single row as vector.
row :: (MatrixElem a, V.VectorElem a) => Int -> Matrix a -> V.Vector a
row i m = V.fromList [m ! (i, j) | j <- [0..cols m - 1]]

-- | Extract single column as vector.
col :: (MatrixElem a, V.VectorElem a) => Int -> Matrix a -> V.Vector a
col j m = V.fromList [m ! (i, j) | i <- [0..rows m - 1]]

-- | Get row (alias for row).
getRow :: (MatrixElem a, V.VectorElem a) => Int -> Matrix a -> V.Vector a
getRow = row

-- | Get column (alias for col).
getCol :: (MatrixElem a, V.VectorElem a) => Int -> Matrix a -> V.Vector a
getCol = col

-- | Get main diagonal.
getDiag :: (MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
getDiag m = V.fromList [m ! (i, i) | i <- [0.. P.min (rows m) (cols m) - 1]]

-- ============================================================
-- Slicing
-- ============================================================

-- | Extract submatrix.
--
-- >>> submatrix 0 2 0 2 m  -- 2x2 top-left corner
submatrix :: MatrixElem a => Int -> Int -> Int -> Int -> Matrix a -> Matrix a
submatrix r1 r2 c1 c2 m =
    let newRows = r2 - r1
        newCols = c2 - c1
    in fromList newRows newCols [m ! (i, j) | i <- [r1..r2-1], j <- [c1..c2-1]]

-- | Take first n rows.
takeRows :: MatrixElem a => Int -> Matrix a -> Matrix a
takeRows n m = submatrix 0 n 0 (cols m) m

-- | Drop first n rows.
dropRows :: MatrixElem a => Int -> Matrix a -> Matrix a
dropRows n m = submatrix n (rows m) 0 (cols m) m

-- | Take first n columns.
takeCols :: MatrixElem a => Int -> Matrix a -> Matrix a
takeCols n m = submatrix 0 (rows m) 0 n m

-- | Drop first n columns.
dropCols :: MatrixElem a => Int -> Matrix a -> Matrix a
dropCols n m = submatrix 0 (rows m) n (cols m) m

-- ============================================================
-- Combining
-- ============================================================

-- | Horizontal concatenation.
--
-- >>> a ||| b
-- [a b]
(|||) :: MatrixElem a => Matrix a -> Matrix a -> Matrix a
a ||| b =
    let r = rows a
        c1 = cols a
        c2 = cols b
    in if rows a /= rows b
       then error "Row counts must match for horizontal concatenation"
       else fromList r (c1 + c2)
            [if j < c1 then a ! (i, j) else b ! (i, j - c1)
             | i <- [0..r-1], j <- [0..c1+c2-1]]
infixr 5 |||

-- | Vertical concatenation.
--
-- >>> a --- b
-- [a]
-- [b]
(---) :: MatrixElem a => Matrix a -> Matrix a -> Matrix a
a --- b =
    let r1 = rows a
        r2 = rows b
        c = cols a
    in if cols a /= cols b
       then error "Column counts must match for vertical concatenation"
       else fromList (r1 + r2) c
            [if i < r1 then a ! (i, j) else b ! (i - r1, j)
             | i <- [0..r1+r2-1], j <- [0..c-1]]
infixr 4 ---

-- | Horizontal concatenation of multiple matrices.
hcat :: MatrixElem a => [Matrix a] -> Matrix a
hcat = P.foldl1 (|||)

-- | Vertical concatenation of multiple matrices.
vcat :: MatrixElem a => [Matrix a] -> Matrix a
vcat = P.foldl1 (---)

-- | Create matrix from blocks.
--
-- >>> fromBlocks [[a, b], [c, d]]
fromBlocks :: MatrixElem a => [[Matrix a]] -> Matrix a
fromBlocks blocks = vcat (P.map hcat blocks)

-- ============================================================
-- Element-wise Operations
-- ============================================================

-- | Map function over elements.
map :: (MatrixElem a, MatrixElem b) => (a -> b) -> Matrix a -> Matrix b
map f m = fromList (rows m) (cols m)
    [f (m ! (i, j)) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Map with indices.
imap :: (MatrixElem a, MatrixElem b) => (Int -> Int -> a -> b) -> Matrix a -> Matrix b
imap f m = fromList (rows m) (cols m)
    [f i j (m ! (i, j)) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Zip two matrices.
zipWith :: (MatrixElem a, MatrixElem b, MatrixElem c) => (a -> b -> c) -> Matrix a -> Matrix b -> Matrix c
zipWith f ma mb =
    let r = P.min (rows ma) (rows mb)
        c = P.min (cols ma) (cols mb)
    in fromList r c
       [f (ma ! (i, j)) (mb ! (i, j)) | i <- [0..r-1], j <- [0..c-1]]

-- | Zip with indices.
izipWith :: (MatrixElem a, MatrixElem b, MatrixElem c) => (Int -> Int -> a -> b -> c) -> Matrix a -> Matrix b -> Matrix c
izipWith f ma mb =
    let r = P.min (rows ma) (rows mb)
        c = P.min (cols ma) (cols mb)
    in fromList r c
       [f i j (ma ! (i, j)) (mb ! (i, j)) | i <- [0..r-1], j <- [0..c-1]]

-- ============================================================
-- Matrix Operations
-- ============================================================

-- | Transpose matrix.
--
-- >>> transpose m
-- m^T
transpose :: MatrixElem a => Matrix a -> Matrix a
transpose m = unsafePerformIO $ matrixTranspose m
{-# NOINLINE transpose #-}

-- | Transpose operator.
(^.) :: MatrixElem a => Matrix a -> Matrix a
(^.) = transpose
infixl 8 ^.

-- | Element-wise addition.
(+.) :: (Num a, MatrixElem a) => Matrix a -> Matrix a -> Matrix a
(+.) = zipWith (+)
infixl 6 +.

-- | Element-wise subtraction.
(-.) :: (Num a, MatrixElem a) => Matrix a -> Matrix a -> Matrix a
(-.) = zipWith (-)
infixl 6 -.

-- | Element-wise multiplication (Hadamard product).
(*.) :: (Num a, MatrixElem a) => Matrix a -> Matrix a -> Matrix a
(*.) = zipWith (*)
infixl 7 *.

-- | Element-wise division.
(/.) :: (Fractional a, MatrixElem a) => Matrix a -> Matrix a -> Matrix a
(/.) = zipWith (/)
infixl 7 /.

-- | Scale matrix by scalar.
scale :: MatrixElem a => a -> Matrix a -> Matrix a
scale k m = unsafePerformIO $ matrixScale k m
{-# NOINLINE scale #-}

-- ============================================================
-- Matrix Multiplication
-- ============================================================

-- | Matrix multiplication.
--
-- ==== __Complexity__
--
-- O(n * m * k) for (n x m) @@ (m x k)
--
-- Uses BLAS DGEMM when available.
mul :: MatrixElem a => Matrix a -> Matrix a -> Matrix a
mul a b = unsafePerformIO $ matrixMatmul a b
{-# NOINLINE mul #-}

-- | Matrix multiplication operator.
(@@) :: MatrixElem a => Matrix a -> Matrix a -> Matrix a
(@@) = mul
infixl 7 @@

-- | Matrix-vector multiplication.
--
-- >>> mulV m v  -- m @ v
mulV :: (Num a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a -> V.Vector a
mulV m v =
    let r = rows m
        c = cols m
    in V.fromList [P.sum [m ! (i, j) * (v V.! j) | j <- [0..c-1]] | i <- [0..r-1]]

-- ============================================================
-- Linear Algebra
-- ============================================================

-- | Matrix trace (sum of diagonal).
trace :: MatrixElem a => Matrix a -> a
trace m = unsafePerformIO $ matrixTrace m
{-# NOINLINE trace #-}

-- | Matrix determinant.
-- Note: Currently a placeholder, full implementation requires LU decomposition
det :: (Num a, MatrixElem a) => Matrix a -> a
det m
    | rows m /= cols m = error "Determinant requires square matrix"
    | rows m == 1 = m ! (0, 0)
    | rows m == 2 = m ! (0, 0) * m ! (1, 1) - m ! (0, 1) * m ! (1, 0)
    | otherwise = error "Determinant for n>2 requires LU decomposition (not yet implemented)"

-- | Matrix rank.
-- Note: Currently a placeholder, requires SVD for robust implementation
rank :: (Ord a, Num a, MatrixElem a) => Matrix a -> Int
rank m = error "Matrix rank requires SVD (not yet implemented)"

-- | Matrix inverse.
--
-- Throws error if matrix is singular.
-- Note: Currently a placeholder, requires LU decomposition
inv :: (Fractional a, MatrixElem a) => Matrix a -> Matrix a
inv m
    | rows m /= cols m = error "Inverse requires square matrix"
    | otherwise = error "Matrix inverse requires LU decomposition (not yet implemented)"

-- | Moore-Penrose pseudoinverse.
-- Note: Currently a placeholder, requires SVD
pinv :: (Floating a, MatrixElem a) => Matrix a -> Matrix a
pinv m = error "Pseudoinverse requires SVD (not yet implemented)"

-- | Solve linear system Ax = b.
--
-- Returns x such that Ax = b.
-- Note: Currently a placeholder, requires LU decomposition
solve :: (Fractional a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a -> V.Vector a
solve a b = error "Linear solve requires LU decomposition (not yet implemented)"

-- | Least squares solution.
--
-- Minimizes ||Ax - b||_2.
-- Note: Currently a placeholder, requires QR or SVD
lstsq :: (Floating a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a -> V.Vector a
lstsq a b = error "Least squares requires QR/SVD (not yet implemented)"

-- ============================================================
-- Decompositions
-- ============================================================

-- | LU decomposition.
--
-- Returns (L, U, P) where PA = LU.
-- Note: Placeholder, requires full implementation
lu :: (Floating a, MatrixElem a) => Matrix a -> (Matrix a, Matrix a, Matrix a)
lu m = error "LU decomposition not yet implemented"

-- | QR decomposition.
--
-- Returns (Q, R) where A = QR.
-- Note: Placeholder, requires Householder or Gram-Schmidt
qr :: (Floating a, MatrixElem a) => Matrix a -> (Matrix a, Matrix a)
qr m = error "QR decomposition not yet implemented"

-- | Singular value decomposition.
--
-- Returns (U, S, V) where A = U * diag(S) * V^T.
-- Note: Placeholder, requires iterative algorithm
svd :: (Floating a, MatrixElem a, V.VectorElem a) => Matrix a -> (Matrix a, V.Vector a, Matrix a)
svd m = error "SVD not yet implemented"

-- | Cholesky decomposition.
--
-- Returns L where A = LL^T.
-- Requires A to be positive definite.
-- Note: Placeholder
cholesky :: (Floating a, MatrixElem a) => Matrix a -> Matrix a
cholesky m = error "Cholesky decomposition not yet implemented"

-- | Eigendecomposition.
--
-- Returns (eigenvalues, eigenvectors).
-- Eigenvectors are columns of the matrix.
-- Note: Placeholder
eig :: (Floating a, MatrixElem a, V.VectorElem (Complex a), MatrixElem (Complex a)) => Matrix a -> (V.Vector (Complex a), Matrix (Complex a))
eig m = error "Eigendecomposition not yet implemented"

-- | Eigenvalues only.
-- Note: Placeholder
eigvals :: (Floating a, MatrixElem a, V.VectorElem (Complex a)) => Matrix a -> V.Vector (Complex a)
eigvals m = error "Eigenvalues not yet implemented"

-- Complex number placeholder
data Complex a = Complex !a !a
    deriving (Eq, Show)

-- ============================================================
-- Norms
-- ============================================================

-- | 1-norm (maximum column sum).
norm1 :: (Num a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> a
norm1 m = P.maximum [V.sum (V.map P.abs (col j m)) | j <- [0..cols m - 1]]

-- | 2-norm (spectral norm, largest singular value).
-- Note: Requires SVD for correct implementation, using Frobenius as approximation
norm2 :: MatrixElem a => Matrix a -> a
norm2 m = unsafePerformIO $ matrixNorm m
{-# NOINLINE norm2 #-}

-- | Infinity norm (maximum row sum).
normInf :: (Num a, Ord a, MatrixElem a, V.VectorElem a) => Matrix a -> a
normInf m = P.maximum [V.sum (V.map P.abs (row i m)) | i <- [0..rows m - 1]]

-- | Frobenius norm (sqrt of sum of squares).
normFrob :: (Floating a, MatrixElem a) => Matrix a -> a
normFrob m = P.sqrt (P.sum [m ! (i, j) ^ (2 :: Int) | i <- [0..rows m - 1], j <- [0..cols m - 1]])

-- ============================================================
-- Folds
-- ============================================================

-- | Sum of all elements.
sum :: (Num a, MatrixElem a) => Matrix a -> a
sum m = P.sum [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Product of all elements.
product :: (Num a, MatrixElem a) => Matrix a -> a
product m = P.product [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Maximum element.
maximum :: (Ord a, MatrixElem a) => Matrix a -> a
maximum m = P.maximum [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Minimum element.
minimum :: (Ord a, MatrixElem a) => Matrix a -> a
minimum m = P.minimum [m ! (i, j) | i <- [0..rows m - 1], j <- [0..cols m - 1]]

-- | Sum each row.
sumRows :: (Num a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
sumRows m = V.generate (rows m) (\i -> V.sum (row i m))

-- | Sum each column.
sumCols :: (Num a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
sumCols m = V.generate (cols m) (\j -> V.sum (col j m))

-- | Mean of each row.
meanRows :: (Fractional a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
meanRows m = V.map (/ P.fromIntegral (cols m)) (sumRows m)

-- | Mean of each column.
meanCols :: (Fractional a, MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
meanCols m = V.map (/ P.fromIntegral (rows m)) (sumCols m)

-- ============================================================
-- Conversion
-- ============================================================

-- | Convert to nested lists.
toLists :: MatrixElem a => Matrix a -> [[a]]
toLists m = [[m ! (i, j) | j <- [0..cols m - 1]] | i <- [0..rows m - 1]]

-- | Convert to flat list (row-major).
toList :: MatrixElem a => Matrix a -> [a]
toList = P.concat . toLists

-- | Flatten to vector.
flatten :: (MatrixElem a, V.VectorElem a) => Matrix a -> V.Vector a
flatten m = V.fromList (toList m)

-- | View vector as 1-column matrix.
asColumn :: (V.VectorElem a, MatrixElem a) => V.Vector a -> Matrix a
asColumn v = fromList (V.length v) 1 (V.toList v)

-- | View vector as 1-row matrix.
asVector :: (V.VectorElem a, MatrixElem a) => V.Vector a -> Matrix a
asVector v = fromList 1 (V.length v) (V.toList v)
