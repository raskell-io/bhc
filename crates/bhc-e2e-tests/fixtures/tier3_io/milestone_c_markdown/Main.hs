module Main where

import Types
import Parser
import Render

main = do
  let blocks = parseMarkdown "# Hello\n\nA paragraph.\n"
  let html = renderDocument blocks
  putStrLn html
