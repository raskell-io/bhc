-- Milestone E: JSON-like key-value parsing
-- Demonstrates parsing a simple key-value format: {"key": "value", "num": 42}
-- Uses separate association lists for string values and int values
-- Avoids Maybe type to work around codegen limitations
-- Uses nested pairs (a, (b, c)) instead of 3-tuples to avoid pattern match issues

-- String equality
stringEq :: String -> String -> Bool
stringEq s1 s2 = case s1 of
    [] -> case s2 of
        [] -> True
        (_:_) -> False
    (c1:cs1) -> case s2 of
        [] -> False
        (c2:cs2) -> (c1 == c2) && stringEq cs1 cs2

-- Lookup in String -> String map, returns "" if not found
lookupStr :: String -> [(String, String)] -> String
lookupStr key kvs = case kvs of
    [] -> ""
    ((k, v) : rest) ->
        if stringEq key k
            then v
            else lookupStr key rest

-- Lookup in String -> Int map, returns -1 if not found
lookupIntVal :: String -> [(String, Int)] -> Int
lookupIntVal key kvs = case kvs of
    [] -> -1
    ((k, v) : rest) ->
        if stringEq key k
            then v
            else lookupIntVal key rest

-- Drop leading whitespace
dropSpaces :: String -> String
dropSpaces s = case s of
    [] -> []
    (c:cs) ->
        if c == ' ' || c == '\n' || c == '\t'
            then dropSpaces cs
            else c : cs

-- Check if character is a digit
digitCheck :: Char -> Bool
digitCheck c = case c of
    '0' -> True
    '1' -> True
    '2' -> True
    '3' -> True
    '4' -> True
    '5' -> True
    '6' -> True
    '7' -> True
    '8' -> True
    '9' -> True
    _ -> False

-- Convert digit char to int
digitVal :: Char -> Int
digitVal c = case c of
    '0' -> 0
    '1' -> 1
    '2' -> 2
    '3' -> 3
    '4' -> 4
    '5' -> 5
    '6' -> 6
    '7' -> 7
    '8' -> 8
    '9' -> 9
    _ -> 0

-- Parse digits, returns (number, rest)
-- Note: Uses let binding to work around expression evaluation issue
parseNum :: Int -> String -> (Int, String)
parseNum acc s = case s of
    [] -> (acc, [])
    (c:cs) ->
        if digitCheck c
            then let newAcc = acc * 10 + digitVal c
                 in parseNum newAcc cs
            else (acc, c:cs)

-- Parse string content after opening quote
-- Note: Uses if-then-else instead of multiple cons patterns to avoid
-- codegen issue with duplicate constructor tags in case alternatives
parseStrContent :: String -> (String, String)
parseStrContent s = case s of
    [] -> ([], [])
    (c:rest) ->
        if c == '"'
            then ([], rest)
            else let result = parseStrContent rest
                 in (c : fst result, snd result)

-- Parse a string value starting at '"', returns (value, rest) or ("", original) on failure
parseStrVal :: String -> (String, String)
parseStrVal s = case s of
    (c:rest) ->
        if c == '"'
            then parseStrContent rest
            else ("", s)
    [] -> ("", s)

-- Parse an int value, returns (value, rest) or (-1, original) on failure
parseIntVal :: String -> (Int, String)
parseIntVal s = case s of
    [] -> (-1, [])
    (c:_) ->
        if digitCheck c
            then parseNum 0 s
            else (-1, s)

-- Get the first string field: returns (key, (value, rest)) or ("", ("", original))
-- Uses nested pair to avoid 3-tuple pattern matching issues
parseStrField :: String -> (String, (String, String))
parseStrField s =
    let s1 = dropSpaces s
    in case s1 of
        (c:rest) ->
            if c == '"'
                then let keyResult = parseStrContent rest
                         s2 = dropSpaces (snd keyResult)
                     in case s2 of
                         (ch:s3) ->
                             if ch == ':'
                                 then let s4 = dropSpaces s3
                                      in case s4 of
                                          (d:_) ->
                                              if d == '"'
                                                  then let valResult = parseStrVal s4
                                                       in (fst keyResult, (fst valResult, snd valResult))
                                                  else ("", ("", s1))
                                          [] -> ("", ("", s1))
                                 else ("", ("", s1))
                         [] -> ("", ("", s1))
                else ("", ("", s1))
        [] -> ("", ("", s1))

-- Get the first int field: returns (key, (value, rest)) or ("", (-1, original))
parseIntField :: String -> (String, (Int, String))
parseIntField s =
    let s1 = dropSpaces s
    in case s1 of
        (c:rest) ->
            if c == '"'
                then let keyResult = parseStrContent rest
                         s2 = dropSpaces (snd keyResult)
                     in case s2 of
                         (ch:s3) ->
                             if ch == ':'
                                 then let s4 = dropSpaces s3
                                      in case s4 of
                                          (d:_) ->
                                              if digitCheck d
                                                  then let valResult = parseNum 0 s4
                                                       in (fst keyResult, (fst valResult, snd valResult))
                                                  else ("", (-1, s1))
                                          [] -> ("", (-1, s1))
                                 else ("", (-1, s1))
                         [] -> ("", (-1, s1))
                else ("", (-1, s1))
        [] -> ("", (-1, s1))

-- Skip one field and return rest
-- Note: Uses if-then-else to avoid duplicate cons pattern issue
skipField :: String -> String
skipField s =
    let s1 = dropSpaces s
    in case s1 of
        (c:rest) ->
            if c == '"'
                then let keyResult = parseStrContent rest
                         s2 = dropSpaces (snd keyResult)
                     in case s2 of
                         (ch:s3) ->
                             if ch == ':'
                                 then let s4 = dropSpaces s3
                                      in case s4 of
                                          [] -> []
                                          (d:ds) ->
                                              if d == '"'
                                                  then let valResult = parseStrContent ds
                                                       in snd valResult
                                                  else if digitCheck d
                                                      then snd (parseNum 0 s4)
                                                      else s4
                                 else s2
                         [] -> s2
                else s1
        [] -> s1

-- Find and parse int field with given key
findIntField :: String -> String -> Int
findIntField key s =
    let s1 = dropSpaces s
    in case s1 of
        [] -> -1
        (c:_) ->
            if c == '}'
                then -1
                else let result = parseIntField s1
                         rKey = fst result
                         rVal = fst (snd result)
                         rRest = snd (snd result)
                     in if stringEq rKey key
                         then rVal
                         else if stringEq rKey ""
                             then -- Not an int field, skip this field and continue
                                 let s2 = skipField s1
                                     s3 = dropSpaces s2
                                 in case s3 of
                                     (ch:s4) ->
                                         if ch == ','
                                             then findIntField key s4
                                             else -1
                                     [] -> -1
                             else -- Found an int field but not the one we want
                                 let s2 = dropSpaces rRest
                                 in case s2 of
                                     (ch:s3) ->
                                         if ch == ','
                                             then findIntField key s3
                                             else -1
                                     [] -> -1

-- Find and parse string field with given key
findStrField :: String -> String -> String
findStrField key s =
    let s1 = dropSpaces s
    in case s1 of
        [] -> ""
        (c:_) ->
            if c == '}'
                then ""
                else let result = parseStrField s1
                         rKey = fst result
                         rVal = fst (snd result)
                         rRest = snd (snd result)
                     in if stringEq rKey key
                         then rVal
                         else if stringEq rKey ""
                             then -- Not a string field, skip and continue
                                 let s2 = skipField s1
                                     s3 = dropSpaces s2
                                 in case s3 of
                                     (ch:s4) ->
                                         if ch == ','
                                             then findStrField key s4
                                             else ""
                                     [] -> ""
                             else -- Found a string field but not the one we want
                                 let s2 = dropSpaces rRest
                                 in case s2 of
                                     (ch:s3) ->
                                         if ch == ','
                                             then findStrField key s3
                                             else ""
                                     [] -> ""

-- Convert Int digit to Char
intToChar :: Int -> Char
intToChar d = case d of
    0 -> '0'
    1 -> '1'
    2 -> '2'
    3 -> '3'
    4 -> '4'
    5 -> '5'
    6 -> '6'
    7 -> '7'
    8 -> '8'
    9 -> '9'
    _ -> '?'

-- Show helper for Int conversion
showHelper :: Int -> String -> String
showHelper x acc = case x of
    0 -> acc
    _ ->
        let d = x `mod` 10
            c = intToChar d
        in showHelper (x `div` 10) (c : acc)

-- Show Int as String
showNum :: Int -> String
showNum n =
    if n == 0
        then "0"
        else showHelper n []

main :: IO ()
main = do
    let json = "{\"name\": \"Alice\", \"age\": 30}"
    -- Skip the opening brace
    let inside = case json of
            (c:rest) ->
                if c == '{'
                    then rest
                    else ""
            [] -> ""
    let name = findStrField "name" inside
    let age = findIntField "age" inside
    putStrLn name
    putStrLn (showNum age)
