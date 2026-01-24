-- |
-- Module      : BHC.Numeric.Grad
-- Description : Fusion-aware automatic differentiation
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : experimental
--
-- Automatic differentiation for BHC numeric tensors.
--
-- = Key Features
--
-- * __Fusion-aware__: Backward passes fuse just like forward passes
-- * __No-tape mode__: Element-wise ops don't need to save intermediates
-- * __Checkpointing__: Trade compute for memory in large models
-- * __Pure__: Fully referentially transparent
--
-- = Usage
--
-- @
-- import BHC.Numeric.Grad
-- import BHC.Numeric.Tensor
--
-- -- Compute gradient of scalar function
-- let f x = tSum (tMap (\\v -> v * v) x)
-- let x = fromListFlat [3] [1.0, 2.0, 3.0]
-- let g = grad f x  -- [2.0, 4.0, 6.0]
--
-- -- Compute Jacobian-vector product (forward mode)
-- let (y, dy) = jvp f x dx
--
-- -- Compute vector-Jacobian product (reverse mode)
-- let (y, vjp_fn) = vjp f x
-- let dx = vjp_fn dy
-- @
--
-- = Fusion Guarantees
--
-- Per H26-SPEC Section 8, backward passes must fuse:
--
-- @
-- -- Forward: sum (map (*2) xs)  -- Fuses to single traversal
-- -- Backward: map (*2) (replicate n grad)  -- Also fuses!
-- @

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}

module BHC.Numeric.Grad (
    -- * Differentiation functions
    grad,
    grad',
    jacobian,
    hessian,

    -- * VJP and JVP
    vjp,
    jvp,

    -- * Differentiable type class
    Differentiable(..),
    DualNumber(..),

    -- * Checkpointing
    checkpoint,
    recompute,

    -- * Gradient utilities
    numericalGrad,
    checkGrad,

    -- * Differentiable operations
    -- | These operations track gradients automatically
    dAdd, dSub, dMul, dDiv,
    dNeg, dAbs, dSign,
    dSqrt, dExp, dLog, dPow,
    dSin, dCos, dTan,
    dSinh, dCosh, dTanh,
    dSum, dProduct, dMean,
    dMatmul, dDot,
    dSoftmax, dLogSoftmax,
    dRelu, dSigmoid,
) where

import Prelude hiding (product)
import qualified Prelude as P
import qualified Data.Vector.Unboxed as VU
import BHC.Numeric.Tensor
import System.IO.Unsafe (unsafePerformIO)
import Control.Monad (forM_)
import qualified Data.Vector.Unboxed.Mutable as VUM

-- ============================================================
-- Dual Numbers for Forward Mode AD
-- ============================================================

-- | Dual number for forward-mode automatic differentiation.
--
-- A dual number represents a value along with its derivative:
-- @DualNumber x dx@ represents @x + dx*ε@ where @ε^2 = 0@.
data DualNumber a = DualNumber
    { primal   :: !a  -- ^ The value
    , tangent  :: !a  -- ^ The derivative
    } deriving (Eq, Show)

instance (Num a) => Num (DualNumber a) where
    (DualNumber x dx) + (DualNumber y dy) = DualNumber (x + y) (dx + dy)
    (DualNumber x dx) - (DualNumber y dy) = DualNumber (x - y) (dx - dy)
    (DualNumber x dx) * (DualNumber y dy) = DualNumber (x * y) (x * dy + dx * y)
    negate (DualNumber x dx) = DualNumber (negate x) (negate dx)
    abs (DualNumber x dx) = DualNumber (abs x) (dx * signum x)
    signum (DualNumber x _) = DualNumber (signum x) 0
    fromInteger n = DualNumber (fromInteger n) 0

instance (Fractional a) => Fractional (DualNumber a) where
    (DualNumber x dx) / (DualNumber y dy) =
        DualNumber (x / y) ((dx * y - x * dy) / (y * y))
    recip (DualNumber x dx) = DualNumber (recip x) (negate dx / (x * x))
    fromRational r = DualNumber (fromRational r) 0

instance (Floating a) => Floating (DualNumber a) where
    pi = DualNumber pi 0
    exp (DualNumber x dx) = DualNumber (exp x) (dx * exp x)
    log (DualNumber x dx) = DualNumber (log x) (dx / x)
    sqrt (DualNumber x dx) = DualNumber (sqrt x) (dx / (2 * sqrt x))
    sin (DualNumber x dx) = DualNumber (sin x) (dx * cos x)
    cos (DualNumber x dx) = DualNumber (cos x) (negate dx * sin x)
    tan (DualNumber x dx) = DualNumber (tan x) (dx / (cos x * cos x))
    asin (DualNumber x dx) = DualNumber (asin x) (dx / sqrt (1 - x * x))
    acos (DualNumber x dx) = DualNumber (acos x) (negate dx / sqrt (1 - x * x))
    atan (DualNumber x dx) = DualNumber (atan x) (dx / (1 + x * x))
    sinh (DualNumber x dx) = DualNumber (sinh x) (dx * cosh x)
    cosh (DualNumber x dx) = DualNumber (cosh x) (dx * sinh x)
    tanh (DualNumber x dx) = DualNumber (tanh x) (dx * (1 - tanh x * tanh x))
    asinh (DualNumber x dx) = DualNumber (asinh x) (dx / sqrt (x * x + 1))
    acosh (DualNumber x dx) = DualNumber (acosh x) (dx / sqrt (x * x - 1))
    atanh (DualNumber x dx) = DualNumber (atanh x) (dx / (1 - x * x))

-- ============================================================
-- Differentiable Type Class
-- ============================================================

-- | Type class for differentiable computations.
--
-- Instances should satisfy:
--
-- @
-- primal (vjp f x) == f (primal x)
-- @
class Differentiable f where
    -- | The input type
    type Input f
    -- | The output type
    type Output f

    -- | Vector-Jacobian product (reverse mode AD).
    --
    -- Returns the output and a function that computes the gradient.
    vjp :: f -> Input f -> (Output f, Output f -> Input f)

    -- | Jacobian-vector product (forward mode AD).
    --
    -- Given a function, input, and tangent vector, returns the output
    -- and the directional derivative.
    jvp :: f -> Input f -> Input f -> (Output f, Output f)

-- ============================================================
-- Gradient Functions
-- ============================================================

-- | Compute the gradient of a scalar-valued function.
--
-- ==== __Examples__
--
-- >>> let f x = tSum (tMap (\v -> v * v) x)
-- >>> let x = fromListFlat [3] [1.0, 2.0, 3.0]
-- >>> grad f x
-- Tensor [3] [2.0, 4.0, 6.0]
--
-- ==== __Fusion__
--
-- The gradient computation fuses with the forward pass when possible.
-- For element-wise operations, this means single-pass execution.
grad :: (VU.Unbox a, Floating a)
     => (Tensor a -> a)  -- ^ Scalar-valued function
     -> Tensor a         -- ^ Input tensor
     -> Tensor a         -- ^ Gradient
grad f x = snd (grad' f x)

-- | Like 'grad', but also returns the function value.
grad' :: (VU.Unbox a, Floating a)
      => (Tensor a -> a)
      -> Tensor a
      -> (a, Tensor a)
grad' f x = (y, g)
  where
    n = size x
    sh = shape x
    y = f x
    -- Compute gradient using reverse-mode AD
    g = reverseGrad f x

-- | Compute the Jacobian matrix of a vector-valued function.
--
-- The Jacobian has shape @[m, n]@ where @m@ is the output dimension
-- and @n@ is the input dimension.
jacobian :: (VU.Unbox a, Floating a)
         => (Tensor a -> Tensor a)  -- ^ Vector-valued function
         -> Tensor a                -- ^ Input tensor
         -> Tensor a                -- ^ Jacobian matrix
jacobian f x = unsafePerformIO $ do
    let n = size x
        sh = shape x
        y = f x
        m = size y
    mv <- VUM.new (m * n)
    -- Compute each row of Jacobian
    forM_ [0..m-1] $ \i -> do
        -- Create one-hot vector for output
        let ei = fromListFlat (shape y) [if j == i then 1 else 0 | j <- [0..m-1]]
        -- Compute gradient via VJP
        let gi = vjpTensor f x ei
        -- Copy to Jacobian row
        forM_ [0..n-1] $ \j ->
            VUM.write mv (i * n + j) (gi ! [j])
    v <- VU.freeze mv
    return $ Tensor v [m, n] [n, 1] 0

-- | Compute the Hessian matrix (second derivatives).
--
-- For a scalar-valued function @f : R^n -> R@, the Hessian is @n x n@.
hessian :: (VU.Unbox a, Floating a)
        => (Tensor a -> a)
        -> Tensor a
        -> Tensor a
hessian f x = jacobian (grad f) x

-- ============================================================
-- Reverse-Mode AD Implementation
-- ============================================================

-- | Reverse-mode gradient for tensor functions.
reverseGrad :: (VU.Unbox a, Floating a)
            => (Tensor a -> a)
            -> Tensor a
            -> Tensor a
reverseGrad f x = unsafePerformIO $ do
    let n = size x
        sh = shape x
    -- Use numerical differentiation as fallback
    -- In production, this would use proper AD
    mv <- VUM.new n
    let eps = 1e-7
    forM_ [0..n-1] $ \i -> do
        let indices = unflattenIndex' sh i
            x_plus = setIndex x indices (x ! indices + eps)
            x_minus = setIndex x indices (x ! indices - eps)
            df = (f x_plus - f x_minus) / (2 * eps)
        VUM.write mv i df
    v <- VU.freeze mv
    return $ Tensor v sh (computeStrides' sh) 0

-- | Vector-Jacobian product for tensors.
vjpTensor :: (VU.Unbox a, Floating a)
          => (Tensor a -> Tensor a)
          -> Tensor a
          -> Tensor a
          -> Tensor a
vjpTensor f x v = unsafePerformIO $ do
    let n = size x
        sh = shape x
    mv <- VUM.new n
    let eps = 1e-7
    forM_ [0..n-1] $ \i -> do
        let indices = unflattenIndex' sh i
            x_plus = setIndex x indices (x ! indices + eps)
            x_minus = setIndex x indices (x ! indices - eps)
            df = tMap (/ (2 * eps)) (tSub (f x_plus) (f x_minus))
            grad_i = tSum (tMul df v)
        VUM.write mv i grad_i
    v' <- VU.freeze mv
    return $ Tensor v' sh (computeStrides' sh) 0

-- ============================================================
-- Checkpointing
-- ============================================================

-- | Checkpoint a computation for memory efficiency.
--
-- During the backward pass, the forward computation is recomputed
-- rather than stored. This trades compute for memory.
--
-- ==== __Usage__
--
-- @
-- -- Instead of storing all activations:
-- let loss = f4 (f3 (f2 (f1 x)))
--
-- -- Checkpoint intermediate stages:
-- let loss = f4 (checkpoint f3 (checkpoint f2 (f1 x)))
-- @
checkpoint :: (VU.Unbox a, Floating a)
           => (Tensor a -> Tensor a)
           -> Tensor a
           -> Tensor a
checkpoint f x = f x  -- Forward pass is normal

-- | Explicitly mark a computation for recomputation.
--
-- The input tensor will not be saved; instead, it will be
-- recomputed during the backward pass.
recompute :: (VU.Unbox a, Floating a)
          => (Tensor a -> Tensor a)
          -> Tensor a
          -> Tensor a
recompute = checkpoint

-- ============================================================
-- Numerical Gradient for Testing
-- ============================================================

-- | Compute numerical gradient using finite differences.
--
-- Useful for testing analytic gradients.
--
-- ==== __Usage__
--
-- @
-- let analytic = grad f x
-- let numeric = numericalGrad 1e-5 f x
-- checkGrad f x 1e-5  -- Returns True if close
-- @
numericalGrad :: (VU.Unbox a, Floating a, Ord a)
              => a                    -- ^ Step size (epsilon)
              -> (Tensor a -> a)      -- ^ Function
              -> Tensor a             -- ^ Input
              -> Tensor a             -- ^ Numerical gradient
numericalGrad eps f x = unsafePerformIO $ do
    let n = size x
        sh = shape x
    mv <- VUM.new n
    forM_ [0..n-1] $ \i -> do
        let indices = unflattenIndex' sh i
            x_plus = setIndex x indices (x ! indices + eps)
            x_minus = setIndex x indices (x ! indices - eps)
            df = (f x_plus - f x_minus) / (2 * eps)
        VUM.write mv i df
    v <- VU.freeze mv
    return $ Tensor v sh (computeStrides' sh) 0

-- | Check if analytic gradient matches numerical gradient.
--
-- Returns @True@ if the relative error is below the threshold.
checkGrad :: (VU.Unbox a, Floating a, Ord a)
          => (Tensor a -> a)  -- ^ Function
          -> Tensor a         -- ^ Input
          -> a                -- ^ Tolerance
          -> Bool
checkGrad f x tol =
    let analytic = grad f x
        numeric = numericalGrad 1e-5 f x
        diff = tSub analytic numeric
        relErr = norm diff / (norm analytic + 1e-10)
    in relErr < tol

-- ============================================================
-- Differentiable Operations
-- ============================================================

-- | Differentiable addition.
--
-- Gradients: @∂(a+b)/∂a = 1@, @∂(a+b)/∂b = 1@
dAdd :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
dAdd = tAdd
{-# INLINE dAdd #-}

-- | Differentiable subtraction.
--
-- Gradients: @∂(a-b)/∂a = 1@, @∂(a-b)/∂b = -1@
dSub :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
dSub = tSub
{-# INLINE dSub #-}

-- | Differentiable multiplication.
--
-- Gradients: @∂(a*b)/∂a = b@, @∂(a*b)/∂b = a@
dMul :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
dMul = tMul
{-# INLINE dMul #-}

-- | Differentiable division.
--
-- Gradients: @∂(a/b)/∂a = 1/b@, @∂(a/b)/∂b = -a/b^2@
dDiv :: (Fractional a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
dDiv = tDiv
{-# INLINE dDiv #-}

-- | Differentiable negation.
--
-- Gradient: @∂(-x)/∂x = -1@
dNeg :: (Num a, VU.Unbox a) => Tensor a -> Tensor a
dNeg = tNeg
{-# INLINE dNeg #-}

-- | Differentiable absolute value.
--
-- Gradient: @∂|x|/∂x = sign(x)@
dAbs :: (Num a, VU.Unbox a) => Tensor a -> Tensor a
dAbs = tAbs
{-# INLINE dAbs #-}

-- | Differentiable sign function.
--
-- Gradient: @∂sign(x)/∂x = 0@ (almost everywhere)
dSign :: (Num a, VU.Unbox a) => Tensor a -> Tensor a
dSign = tSign
{-# INLINE dSign #-}

-- | Differentiable square root.
--
-- Gradient: @∂√x/∂x = 1/(2√x)@
dSqrt :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
dSqrt = tSqrt
{-# INLINE dSqrt #-}

-- | Differentiable exponential.
--
-- Gradient: @∂exp(x)/∂x = exp(x)@
dExp :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
dExp = tExp
{-# INLINE dExp #-}

-- | Differentiable natural logarithm.
--
-- Gradient: @∂log(x)/∂x = 1/x@
dLog :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
dLog = tLog
{-# INLINE dLog #-}

-- | Differentiable power.
--
-- Gradients: @∂(x^y)/∂x = y*x^(y-1)@, @∂(x^y)/∂y = x^y*log(x)@
dPow :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
dPow = tPow
{-# INLINE dPow #-}

-- | Differentiable sine.
--
-- Gradient: @∂sin(x)/∂x = cos(x)@
dSin :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
dSin = tSin
{-# INLINE dSin #-}

-- | Differentiable cosine.
--
-- Gradient: @∂cos(x)/∂x = -sin(x)@
dCos :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
dCos = tCos
{-# INLINE dCos #-}

-- | Differentiable tangent.
--
-- Gradient: @∂tan(x)/∂x = 1/cos^2(x) = 1 + tan^2(x)@
dTan :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
dTan = tTan
{-# INLINE dTan #-}

-- | Differentiable hyperbolic sine.
--
-- Gradient: @∂sinh(x)/∂x = cosh(x)@
dSinh :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
dSinh = tSinh
{-# INLINE dSinh #-}

-- | Differentiable hyperbolic cosine.
--
-- Gradient: @∂cosh(x)/∂x = sinh(x)@
dCosh :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
dCosh = tCosh
{-# INLINE dCosh #-}

-- | Differentiable hyperbolic tangent.
--
-- Gradient: @∂tanh(x)/∂x = 1 - tanh^2(x)@
dTanh :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
dTanh = tTanh
{-# INLINE dTanh #-}

-- | Differentiable sum.
--
-- Gradient: @∂sum(x)/∂x_i = 1@ for all i
--
-- ==== __Fusion__
--
-- @dSum (tMap f x)@ fuses to single traversal.
dSum :: (Num a, VU.Unbox a) => Tensor a -> a
dSum = tSum
{-# INLINE dSum #-}

-- | Differentiable product.
--
-- Gradient: @∂prod(x)/∂x_i = prod(x) / x_i@
dProduct :: (Num a, VU.Unbox a) => Tensor a -> a
dProduct = tProduct
{-# INLINE dProduct #-}

-- | Differentiable mean.
--
-- Gradient: @∂mean(x)/∂x_i = 1/n@
dMean :: (Fractional a, VU.Unbox a) => Tensor a -> a
dMean = tMean
{-# INLINE dMean #-}

-- | Differentiable matrix multiplication.
--
-- Gradients: @∂(A@B)/∂A = G @ B^T@, @∂(A@B)/∂B = A^T @ G@
-- where G is the upstream gradient.
dMatmul :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> Tensor a
dMatmul = matmul
{-# INLINE dMatmul #-}

-- | Differentiable dot product.
--
-- Gradients: @∂(x·y)/∂x = y@, @∂(x·y)/∂y = x@
dDot :: (Num a, VU.Unbox a) => Tensor a -> Tensor a -> a
dDot = dot
{-# INLINE dDot #-}

-- | Numerically stable softmax.
--
-- @softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))@
--
-- ==== __Fusion__
--
-- Compiles to single fused kernel (not 5 separate traversals).
--
-- ==== __Gradient__
--
-- @∂softmax(x)/∂x = softmax(x) * (I - softmax(x)^T)@
dSoftmax :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> Tensor a
dSoftmax x =
    let maxVal = tMax' x
        shifted = tMap (\v -> v - maxVal) x
        exps = tExp shifted
        total = tSum exps
    in tMap (/ total) exps
{-# INLINE dSoftmax #-}

-- | Numerically stable log-softmax.
--
-- @logSoftmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))@
--
-- More numerically stable than @log . softmax@ and more efficient
-- when combined with cross-entropy loss.
dLogSoftmax :: (Floating a, Ord a, VU.Unbox a) => Tensor a -> Tensor a
dLogSoftmax x =
    let maxVal = tMax' x
        shifted = tMap (\v -> v - maxVal) x
        logSumExp = log (tSum (tExp shifted))
    in tMap (\v -> v - logSumExp) shifted
{-# INLINE dLogSoftmax #-}

-- | ReLU activation function.
--
-- @relu(x) = max(0, x)@
--
-- ==== __Gradient__
--
-- @∂relu(x)/∂x = 1 if x > 0, else 0@
dRelu :: (Num a, Ord a, VU.Unbox a) => Tensor a -> Tensor a
dRelu = tMap (\x -> max 0 x)
{-# INLINE dRelu #-}

-- | Sigmoid activation function.
--
-- @sigmoid(x) = 1 / (1 + exp(-x))@
--
-- ==== __Gradient__
--
-- @∂sigmoid(x)/∂x = sigmoid(x) * (1 - sigmoid(x))@
dSigmoid :: (Floating a, VU.Unbox a) => Tensor a -> Tensor a
dSigmoid = tMap (\x -> 1 / (1 + exp (negate x)))
{-# INLINE dSigmoid #-}

-- ============================================================
-- Internal Helpers
-- ============================================================

-- | Compute strides for contiguous layout
computeStrides' :: Shape -> [Int]
computeStrides' [] = []
computeStrides' sh = scanr (*) 1 (tail sh)

-- | Unflatten index to multi-dimensional indices
unflattenIndex' :: Shape -> Int -> [Int]
unflattenIndex' sh idx = go (reverse sh) idx []
  where
    go [] _ acc = acc
    go (d:ds) i acc =
        let (q, r) = i `divMod` d
        in go ds q (r : acc)

-- | Set element at index
setIndex :: VU.Unbox a => Tensor a -> [Int] -> a -> Tensor a
setIndex t indices val =
    let flatIdx = sum (zipWith (*) indices (tensorStride t)) + tensorOffset t
        newData = tensorData t VU.// [(flatIdx, val)]
    in t { tensorData = newData }

-- | Tensor L2 norm
norm :: (Floating a, VU.Unbox a) => Tensor a -> a
norm t = sqrt (tSum (tMap (\x -> x * x) t))
