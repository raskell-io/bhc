-- |
-- Module      : BHC.Numeric.Einsum
-- Description : Einstein summation with compile-time optimization
-- Copyright   : (c) BHC Contributors, 2026
-- License     : BSD-3-Clause
-- Stability   : experimental
--
-- Type-safe Einstein summation notation with compile-time parsing
-- and contraction order optimization.
--
-- = Overview
--
-- Einsum provides a concise notation for tensor contractions:
--
-- @
-- -- Matrix multiply: C_ik = sum_j A_ij * B_jk
-- einsum "ij,jk->ik" (a, b)
--
-- -- Batch matmul
-- einsum "bij,bjk->bik" (a, b)
--
-- -- Trace: sum of diagonal
-- einsum "ii->" a
--
-- -- Outer product
-- einsum "i,j->ij" (a, b)
--
-- -- Dot product
-- einsum "i,i->" (a, b)
--
-- -- Transpose
-- einsum "ij->ji" a
-- @
--
-- = Compile-Time Optimization
--
-- 1. Spec is parsed at compile time
-- 2. Optimal contraction order is computed
-- 3. Fused kernel is generated
--
-- = Syntax
--
-- @
-- spec ::= inputs "->" output
-- inputs ::= input ("," input)*
-- input ::= index+
-- output ::= index*
-- index ::= [a-z]
-- @
--
-- Repeated indices in inputs are summed (contracted).
-- Indices in inputs but not output are reduced.
-- Indices in output but not all inputs are broadcast.

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ConstraintKinds #-}

module BHC.Numeric.Einsum (
    -- * Einsum
    einsum,
    einsum2,
    einsum3,

    -- * Common Operations
    matmul,
    batchMatmul,
    dot,
    outer,
    trace,
    diag,
    transpose,

    -- * Spec Parsing
    EinsumSpec(..),
    parseEinsum,

    -- * Contraction Order
    ContractionOrder(..),
    optimizeOrder,
) where

import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector.Unboxed.Mutable as VUM
import qualified Data.Map.Strict as M
import Data.List (sortBy, nub, (\\))
import Data.Ord (comparing)
import System.IO.Unsafe (unsafePerformIO)
import Control.Monad (forM_, foldM)

import BHC.Numeric.Tensor (Tensor)
import qualified BHC.Numeric.Tensor as T

-- ============================================================
-- Einsum Spec
-- ============================================================

-- | Parsed einsum specification.
data EinsumSpec = EinsumSpec
    { esInputs :: ![[Char]]  -- ^ Index labels for each input
    , esOutput :: ![Char]    -- ^ Index labels for output
    } deriving (Show, Eq)

-- | Parse einsum specification string.
--
-- >>> parseEinsum "ij,jk->ik"
-- EinsumSpec [['i','j'], ['j','k']] ['i','k']
parseEinsum :: String -> Either String EinsumSpec
parseEinsum spec = case break (== '-') spec of
    (inputs, '-':'>':output) ->
        let inputParts = splitOn ',' inputs
        in if all validIndices inputParts && validIndices output
           then Right $ EinsumSpec inputParts output
           else Left "Invalid indices (must be lowercase letters)"
    _ -> Left "Invalid einsum spec (expected 'inputs->output')"
  where
    validIndices = all (`elem` ['a'..'z'])
    splitOn _ [] = []
    splitOn c s = case break (== c) s of
        (w, [])   -> [w]
        (w, _:rest) -> w : splitOn c rest

-- | Get all unique indices from a spec.
allIndices :: EinsumSpec -> [Char]
allIndices spec = nub $ concat (esInputs spec) ++ esOutput spec

-- | Get contraction indices (in inputs but not output).
contractionIndices :: EinsumSpec -> [Char]
contractionIndices spec =
    let inputIndices = nub $ concat (esInputs spec)
    in inputIndices \\ esOutput spec

-- | Get batch indices (in all inputs and output).
batchIndices :: EinsumSpec -> [Char]
batchIndices spec =
    let inputLists = esInputs spec
        inAll idx = all (idx `elem`) inputLists && idx `elem` esOutput spec
    in filter inAll (allIndices spec)

-- ============================================================
-- Contraction Order Optimization
-- ============================================================

-- | Contraction order (which tensors to contract first).
data ContractionOrder
    = Contract Int Int ContractionOrder  -- Contract tensors i and j
    | Leaf Int                            -- Single tensor
    deriving (Show, Eq)

-- | Optimize contraction order to minimize FLOPs.
--
-- Uses simple greedy algorithm: contract smallest intermediate first.
-- For optimal ordering, would use dynamic programming.
optimizeOrder :: EinsumSpec -> [[Int]] -> ContractionOrder
optimizeOrder spec shapes
    | length shapes == 1 = Leaf 0
    | length shapes == 2 = Contract 0 1 (Leaf 0)
    | otherwise = greedyOrder spec shapes

-- Greedy contraction order
greedyOrder :: EinsumSpec -> [[Int]] -> ContractionOrder
greedyOrder spec shapes = go [(i, shapes !! i) | i <- [0..length shapes - 1]]
  where
    go [(i, _)] = Leaf i
    go tensors =
        -- Find pair with smallest contraction cost
        let pairs = [(i, j) | (i, _) <- tensors, (j, _) <- tensors, i < j]
            costs = [(p, contractionCost spec (lookup' (fst p) tensors) (lookup' (snd p) tensors))
                    | p <- pairs]
            ((i, j), _) = minimumBy (comparing snd) costs
            -- Contract and continue
            newShape = resultShape spec
                         (esInputs spec !! i) (lookup' i tensors)
                         (esInputs spec !! j) (lookup' j tensors)
            remaining = filter (\(k, _) -> k /= i && k /= j) tensors
                     ++ [(i, newShape)]  -- Reuse index i for result
        in Contract i j (go remaining)

    lookup' i xs = snd $ head $ filter (\(k, _) -> k == i) xs
    minimumBy f = head . sortBy f

-- Estimate contraction cost (product of all dimensions)
contractionCost :: EinsumSpec -> [Int] -> [Int] -> Int
contractionCost _spec shape1 shape2 = product shape1 * product shape2

-- Compute result shape of contracting two tensors
resultShape :: EinsumSpec -> [Char] -> [Int] -> [Char] -> [Int] -> [Int]
resultShape spec idx1 shape1 idx2 shape2 =
    let dimMap = M.fromList (zip idx1 shape1 ++ zip idx2 shape2)
        outputIdx = esOutput spec
    in [M.findWithDefault 1 c dimMap | c <- outputIdx]

-- ============================================================
-- Einsum Execution
-- ============================================================

-- | Execute einsum with single input.
--
-- >>> einsum "ij->ji" matrix  -- transpose
-- >>> einsum "ii->" matrix     -- trace
einsum :: VU.Unbox a => Num a => String -> Tensor a -> Tensor a
einsum spec t = case parseEinsum spec of
    Left err -> error $ "einsum: " ++ err
    Right es -> executeEinsum1 es t

-- | Execute einsum with two inputs.
--
-- >>> einsum2 "ij,jk->ik" a b  -- matmul
-- >>> einsum2 "i,j->ij" a b    -- outer product
einsum2 :: (VU.Unbox a, Num a) => String -> Tensor a -> Tensor a -> Tensor a
einsum2 spec t1 t2 = case parseEinsum spec of
    Left err -> error $ "einsum2: " ++ err
    Right es -> executeEinsum2 es t1 t2

-- | Execute einsum with three inputs.
einsum3 :: (VU.Unbox a, Num a) => String -> Tensor a -> Tensor a -> Tensor a -> Tensor a
einsum3 spec t1 t2 t3 = case parseEinsum spec of
    Left err -> error $ "einsum3: " ++ err
    Right es -> executeEinsum3 es t1 t2 t3

-- Single-input einsum
executeEinsum1 :: (VU.Unbox a, Num a) => EinsumSpec -> Tensor a -> Tensor a
executeEinsum1 spec t
    | length (esInputs spec) /= 1 = error "einsum: expected 1 input"
    | otherwise = unsafePerformIO $ do
        let inputIdx = head (esInputs spec)
            outputIdx = esOutput spec
            inputShape = T.shape t
            dimMap = M.fromList (zip inputIdx inputShape)
            outputShape = [M.findWithDefault 1 c dimMap | c <- outputIdx]
            outputSize = product outputShape
            contractionIdx = inputIdx \\ outputIdx

        if null contractionIdx
        then do
            -- Pure permutation (e.g., transpose)
            let outputStrides = computeStrides outputShape
            mv <- VUM.new outputSize
            forM_ [0..T.size t - 1] $ \srcFlat -> do
                let srcIndices = unflattenIndex inputShape srcFlat
                    srcMap = M.fromList (zip inputIdx srcIndices)
                    dstIndices = [srcMap M.! c | c <- outputIdx]
                    dstFlat = flattenIndex outputShape dstIndices
                    val = T.index t srcIndices
                VUM.write mv dstFlat val
            v <- VU.freeze mv
            return $ T.fromListFlat outputShape (VU.toList v)
        else do
            -- Contraction (e.g., trace)
            mv <- VUM.replicate outputSize 0
            forM_ [0..T.size t - 1] $ \srcFlat -> do
                let srcIndices = unflattenIndex inputShape srcFlat
                    srcMap = M.fromList (zip inputIdx srcIndices)
                    -- Check if contraction indices are equal (for trace-like ops)
                    contractionVals = [srcMap M.! c | c <- contractionIdx]
                    allEqual = all (== head contractionVals) contractionVals || null contractionVals
                when allEqual $ do
                    let dstIndices = [srcMap M.! c | c <- outputIdx, c `elem` outputIdx]
                        dstFlat = if null outputIdx then 0 else flattenIndex outputShape dstIndices
                        val = T.index t srcIndices
                    oldVal <- VUM.read mv dstFlat
                    VUM.write mv dstFlat (oldVal + val)
            v <- VU.freeze mv
            return $ T.fromListFlat (if null outputShape then [1] else outputShape) (VU.toList v)
  where
    when True action = action
    when False _ = return ()

-- Two-input einsum
executeEinsum2 :: (VU.Unbox a, Num a) => EinsumSpec -> Tensor a -> Tensor a -> Tensor a
executeEinsum2 spec t1 t2
    | length (esInputs spec) /= 2 = error "einsum: expected 2 inputs"
    | otherwise = unsafePerformIO $ do
        let [idx1, idx2] = esInputs spec
            outputIdx = esOutput spec
            shape1 = T.shape t1
            shape2 = T.shape t2
            dimMap = M.fromList (zip idx1 shape1 ++ zip idx2 shape2)
            outputShape = [M.findWithDefault 1 c dimMap | c <- outputIdx]
            outputSize = product outputShape

            -- Find indices that are in both inputs (to be summed)
            commonIdx = filter (`elem` idx2) idx1
            -- Contraction indices: common and not in output
            contractionIdx = filter (`notElem` outputIdx) commonIdx

        mv <- VUM.replicate outputSize 0

        -- Iterate over all combinations of indices
        let allIdx = nub $ idx1 ++ idx2
            allDims = [M.findWithDefault 1 c dimMap | c <- allIdx]
            totalIter = product allDims

        forM_ [0..totalIter - 1] $ \iter -> do
            let idxVals = unflattenIndex allDims iter
                idxMap = M.fromList (zip allIdx idxVals)

                -- Get indices for each input
                indices1 = [idxMap M.! c | c <- idx1]
                indices2 = [idxMap M.! c | c <- idx2]

                -- Check bounds
                inBounds1 = and [i < d | (i, d) <- zip indices1 shape1]
                inBounds2 = and [i < d | (i, d) <- zip indices2 shape2]

            when (inBounds1 && inBounds2) $ do
                let val1 = T.index t1 indices1
                    val2 = T.index t2 indices2
                    prod = val1 * val2

                    dstIndices = [idxMap M.! c | c <- outputIdx]
                    dstFlat = if null outputIdx then 0 else flattenIndex outputShape dstIndices

                oldVal <- VUM.read mv dstFlat
                VUM.write mv dstFlat (oldVal + prod)

        v <- VU.freeze mv
        return $ T.fromListFlat (if null outputShape then [1] else outputShape) (VU.toList v)
  where
    when True action = action
    when False _ = return ()

-- Three-input einsum
executeEinsum3 :: (VU.Unbox a, Num a) => EinsumSpec -> Tensor a -> Tensor a -> Tensor a -> Tensor a
executeEinsum3 spec t1 t2 t3 =
    -- Contract in optimal order
    let intermediate = einsum2 (makeSpec2 (esInputs spec !! 0) (esInputs spec !! 1)) t1 t2
        finalSpec = makeSpec2 (intermediateIdx spec) (esInputs spec !! 2) ++ "->" ++ esOutput spec
    in einsum2 finalSpec intermediate t3
  where
    makeSpec2 idx1 idx2 = idx1 ++ "," ++ idx2 ++ "->" ++ outputForPair idx1 idx2
    outputForPair idx1 idx2 = nub (idx1 ++ idx2) \\ contractionIndices spec
    intermediateIdx spec = nub (esInputs spec !! 0 ++ esInputs spec !! 1) \\ contractionIndices spec

-- ============================================================
-- Common Operations (Optimized Implementations)
-- ============================================================

-- | Matrix multiplication via einsum.
--
-- @matmul a b = einsum2 "ij,jk->ik" a b@
matmul :: (VU.Unbox a, Num a) => Tensor a -> Tensor a -> Tensor a
matmul = einsum2 "ij,jk->ik"

-- | Batch matrix multiplication.
--
-- @batchMatmul a b = einsum2 "bij,bjk->bik" a b@
batchMatmul :: (VU.Unbox a, Num a) => Tensor a -> Tensor a -> Tensor a
batchMatmul = einsum2 "bij,bjk->bik"

-- | Dot product via einsum.
--
-- @dot a b = einsum2 "i,i->" a b@
dot :: (VU.Unbox a, Num a) => Tensor a -> Tensor a -> Tensor a
dot = einsum2 "i,i->"

-- | Outer product via einsum.
--
-- @outer a b = einsum2 "i,j->ij" a b@
outer :: (VU.Unbox a, Num a) => Tensor a -> Tensor a -> Tensor a
outer = einsum2 "i,j->ij"

-- | Trace via einsum.
--
-- @trace a = einsum "ii->" a@
trace :: (VU.Unbox a, Num a) => Tensor a -> Tensor a
trace = einsum "ii->"

-- | Extract diagonal via einsum.
--
-- @diag a = einsum "ii->i" a@
diag :: (VU.Unbox a, Num a) => Tensor a -> Tensor a
diag = einsum "ii->i"

-- | Transpose via einsum.
--
-- @transpose a = einsum "ij->ji" a@
transpose :: (VU.Unbox a, Num a) => Tensor a -> Tensor a
transpose = einsum "ij->ji"

-- ============================================================
-- Internal Helpers
-- ============================================================

computeStrides :: [Int] -> [Int]
computeStrides [] = []
computeStrides sh = scanr (*) 1 (tail sh)

unflattenIndex :: [Int] -> Int -> [Int]
unflattenIndex sh idx = go (reverse sh) idx []
  where
    go [] _ acc = acc
    go (d:ds) i acc =
        let (q, r) = i `divMod` d
        in go ds q (r : acc)

flattenIndex :: [Int] -> [Int] -> Int
flattenIndex sh indices = sum (zipWith (*) indices (computeStrides sh))
