module Parser where

import Types

-- | Split a string on newline characters (replaces builtin `lines`).
splitLines s = case s of
  [] -> []
  _ -> let lineAndRest = splitAtNewline s
       in fst lineAndRest : splitLines (snd lineAndRest)

-- | Split at the first newline, returning (before, after).
splitAtNewline s = case s of
  [] -> ([], [])
  (c : rest) -> case c of
    '\n' -> ([], rest)
    _ -> let result = splitAtNewline rest
         in (c : fst result, snd result)

-- | Parse a markdown string into a list of blocks.
parseMarkdown input = groupLines (splitLines input) [] []

-- | Group lines into blocks.
groupLines ls paraLines blocks = case ls of
  [] -> reverse (flushParagraph paraLines blocks)
  (line : rest) ->
    if isHeadingLine line then
      let level = countHashes line
          text = stripSpaces (drop level line)
      in groupLines rest [] (Heading level text : flushParagraph paraLines blocks)
    else if isHRule line then
      groupLines rest [] (HRule : flushParagraph paraLines blocks)
    else if isCodeFence line then
      let codeAndRest = collectCodeBlock rest
          code = joinLines (fst codeAndRest)
          remaining = snd codeAndRest
      in groupLines remaining [] (CodeBlock code : flushParagraph paraLines blocks)
    else if isUnorderedItem line then
      let itemsAndRest = collectUnorderedList (line : rest)
      in groupLines (snd itemsAndRest) [] (UnorderedList (fst itemsAndRest) : flushParagraph paraLines blocks)
    else if isOrderedItem line then
      let itemsAndRest = collectOrderedList (line : rest)
      in groupLines (snd itemsAndRest) [] (OrderedList (fst itemsAndRest) : flushParagraph paraLines blocks)
    else if isBlockquoteLine line then
      let text = parseBlockquoteText line
      in groupLines rest [] (Blockquote text : flushParagraph paraLines blocks)
    else if isBlankLine line then
      groupLines rest [] (BlankLine : flushParagraph paraLines blocks)
    else
      groupLines rest (paraLines ++ [line]) blocks

-- | Check if a line is blank (empty).
isBlankLine s = case s of
  [] -> True
  _ -> False

-- | Flush accumulated paragraph lines into a Paragraph block.
flushParagraph paraLines blocks = case paraLines of
  [] -> blocks
  _ -> Paragraph (joinWithSpaces paraLines) : blocks

-- | Join strings with spaces.
joinWithSpaces xs = case xs of
  [] -> ""
  (y : ys) -> case ys of
    [] -> y
    _ -> y ++ " " ++ joinWithSpaces ys

-- | Join strings with newlines.
joinLines xs = case xs of
  [] -> ""
  (y : ys) -> case ys of
    [] -> y
    _ -> y ++ "\n" ++ joinLines ys

-- | Strip leading spaces from a string.
stripSpaces s = case s of
  [] -> []
  (c : rest) -> case c of
    ' ' -> stripSpaces rest
    _ -> s

-- | Check if a line is a heading (starts with #).
isHeadingLine s = case s of
  [] -> False
  (c : _) -> case c of
    '#' -> True
    _ -> False

-- | Count leading hash characters.
countHashes s = case s of
  [] -> 0
  (c : rest) -> case c of
    '#' -> 1 + countHashes rest
    _ -> 0

-- | Check if a line is a horizontal rule (---).
isHRule s = case s of
  [] -> False
  (c1 : rest1) -> case c1 of
    '-' -> case rest1 of
      [] -> False
      (c2 : rest2) -> case c2 of
        '-' -> case rest2 of
          [] -> False
          (c3 : _) -> case c3 of
            '-' -> True
            _ -> False
        _ -> False
    _ -> False

-- | Check if a line starts a fenced code block (```).
isCodeFence s = case s of
  [] -> False
  (c1 : rest1) -> case c1 of
    '`' -> case rest1 of
      [] -> False
      (c2 : rest2) -> case c2 of
        '`' -> case rest2 of
          [] -> False
          (c3 : _) -> case c3 of
            '`' -> True
            _ -> False
        _ -> False
    _ -> False

-- | Collect lines until the closing code fence.
collectCodeBlock ls = case ls of
  [] -> ([], [])
  (line : rest) ->
    if isCodeFence line
    then ([], rest)
    else let codeAndRest = collectCodeBlock rest
         in (line : fst codeAndRest, snd codeAndRest)

-- | Check if a line is an unordered list item (- or *).
isUnorderedItem s = case s of
  [] -> False
  (c : rest) -> case c of
    '-' -> case rest of
      [] -> False
      (c2 : _) -> case c2 of
        ' ' -> True
        _ -> False
    '*' -> case rest of
      [] -> False
      (c2 : _) -> case c2 of
        ' ' -> True
        _ -> False
    _ -> False

-- | Parse unordered list item text (drop marker).
parseUnorderedItem s = case s of
  [] -> s
  (c : rest) -> case c of
    '-' -> case rest of
      [] -> s
      (c2 : text) -> case c2 of
        ' ' -> text
        _ -> s
    '*' -> case rest of
      [] -> s
      (c2 : text) -> case c2 of
        ' ' -> text
        _ -> s
    _ -> s

-- | Collect consecutive unordered list items.
collectUnorderedList ls = case ls of
  [] -> ([], [])
  (line : rest) ->
    if isUnorderedItem line
    then let text = parseUnorderedItem line
             itemsAndRest = collectUnorderedList rest
         in (text : fst itemsAndRest, snd itemsAndRest)
    else ([], line : rest)

-- | Check if a line is an ordered list item (digit.space).
isOrderedItem s = case s of
  [] -> False
  (c : rest) -> case c of
    '1' -> checkDotSpace rest
    '2' -> checkDotSpace rest
    '3' -> checkDotSpace rest
    '4' -> checkDotSpace rest
    '5' -> checkDotSpace rest
    '6' -> checkDotSpace rest
    '7' -> checkDotSpace rest
    '8' -> checkDotSpace rest
    '9' -> checkDotSpace rest
    _ -> False

-- | Check for ". " after digit.
checkDotSpace rest = case rest of
  [] -> False
  (c2 : rest2) -> case c2 of
    '.' -> case rest2 of
      [] -> False
      (c3 : _) -> case c3 of
        ' ' -> True
        _ -> False
    _ -> False

-- | Parse ordered list item text (drop "N. ").
parseOrderedItem s = case s of
  [] -> s
  (_ : rest) -> case rest of
    [] -> s
    (c2 : rest2) -> case c2 of
      '.' -> case rest2 of
        [] -> s
        (c3 : text) -> case c3 of
          ' ' -> text
          _ -> s
      _ -> s

-- | Collect consecutive ordered list items.
collectOrderedList ls = case ls of
  [] -> ([], [])
  (line : rest) ->
    if isOrderedItem line
    then let text = parseOrderedItem line
             itemsAndRest = collectOrderedList rest
         in (text : fst itemsAndRest, snd itemsAndRest)
    else ([], line : rest)

-- | Check if a line is a blockquote (starts with >).
isBlockquoteLine s = case s of
  [] -> False
  (c : _) -> case c of
    '>' -> True
    _ -> False

-- | Parse blockquote text (strip leading > and space).
parseBlockquoteText s = case s of
  [] -> s
  (c : rest) -> case c of
    '>' -> case rest of
      [] -> rest
      (c2 : text) -> case c2 of
        ' ' -> text
        _ -> rest
    _ -> s
