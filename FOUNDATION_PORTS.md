# Foundation Port Indexing Reference

Complete visual guide for all foundation types showing port positions.

**Key Concepts:**
- Each 1×1 unit = 14×14 internal grid
- Each external face (edge) has 4 ports (notches)
- Ports are indexed 0, 1, 2, 3... along each side
- 4 floors available (0-3) per port position

---

## 1×1 Foundation (14×14 grid)

**Dimensions:** 1 unit × 1 unit
**Total Ports:** 16 (4 sides × 4 ports/side)

```
        North (N)
      0   1   2   3
    ┌───┬───┬───┬───┐
  0 │               │ 0
W 1 │               │ 1 E
e 2 │    1×1        │ 2 a
s 3 │   14×14       │ 3 s
t   └───┴───┴───┴───┘   t
      0   1   2   3
        South (S)
```

**Port Specification Examples:**
```
W,0,0,CuCuCuCu  # West bottom port
N,2,0,RuRuRuRu  # North port 2
E,3,0,SuSuSuSu  # East top port
S,1,1,WyWyWyWy  # South port 1, floor 1
```

---

## 2×1 Foundation (34×14 grid)

**Dimensions:** 2 units wide × 1 unit tall
**Total Ports:** 24 (N:8 + S:8 + E:4 + W:4)

```
        North (N)
      0 1 2 3 4 5 6 7
    ┌───────┬───────┐
  0 │ Unit0 │ Unit1 │ 0
W 1 │ 14×14 │ 14×14 │ 1 E
e 2 │       │       │ 2 a
s 3 │       │       │ 3 s
t   └───────┴───────┘   t
      0 1 2 3 4 5 6 7
        South (S)
```

**Ports per side:**
- North: 0-7 (8 ports)
- South: 0-7 (8 ports)
- East: 0-3 (4 ports)
- West: 0-3 (4 ports)

---

## 1×2 Foundation (14×34 grid)

**Dimensions:** 1 unit wide × 2 units tall
**Total Ports:** 24 (N:4 + S:4 + E:8 + W:8)

```
      North (N)
      0   1   2   3
    ┌───┬───┬───┬───┐
  0 │               │ 0
  1 │    Unit 0     │ 1
  2 │    14×14      │ 2
  3 │               │ 3
W 4 ├───────────────┤ 4 E
e 5 │               │ 5 a
s 6 │    Unit 1     │ 6 s
t 7 │    14×14      │ 7 t
    └───┴───┴───┴───┘
      0   1   2   3
      South (S)
```

**Ports per side:**
- North: 0-3 (4 ports)
- South: 0-3 (4 ports)
- East: 0-7 (8 ports)
- West: 0-7 (8 ports)

---

## 2×2 Foundation (34×34 grid)

**Dimensions:** 2 units × 2 units
**Total Ports:** 32 (4 sides × 8 ports/side)

```
          North (N)
      0 1 2 3 4 5 6 7
    ┌───────┬───────┐
  0 │ U0,0  │ U1,0  │ 0
  1 │ 14×14 │ 14×14 │ 1
  2 │       │       │ 2
  3 │       │       │ 3
W 4 ├───────┼───────┤ 4 E
e 5 │ U0,1  │ U1,1  │ 5 a
s 6 │ 14×14 │ 14×14 │ 6 s
t 7 │       │       │ 7 t
    └───────┴───────┘
      0 1 2 3 4 5 6 7
          South (S)
```

**Ports per side:** 0-7 on all sides (8 each)

**Example - Corner splitter:**
```
Input:  W,3,0,CuCuCuCu
Output: E,2,0,Cu------
        E,3,0,--Cu----
        E,4,0,----Cu--
        E,5,0,------Cu
```

---

## 3×1 Foundation (54×14 grid)

**Dimensions:** 3 units wide × 1 unit tall
**Total Ports:** 32 (N:12 + S:12 + E:4 + W:4)

```
        North (N)
      0 1 2 3 4 5 6 7 8 9 10 11
    ┌─────┬─────┬─────┐
  0 │ U0  │ U1  │ U2  │ 0
W 1 │14×14│14×14│14×14│ 1 E
e 2 │     │     │     │ 2 a
s 3 │     │     │     │ 3 s
t   └─────┴─────┴─────┘   t
      0 1 2 3 4 5 6 7 8 9 10 11
        South (S)
```

**Ports per side:**
- North: 0-11 (12 ports)
- South: 0-11 (12 ports)
- East: 0-3 (4 ports)
- West: 0-3 (4 ports)

---

## 3×2 Foundation (54×34 grid)

**Dimensions:** 3 units wide × 2 units tall
**Total Ports:** 40 (N:12 + S:12 + E:8 + W:8)

```
          North (N)
      0 1 2 3 4 5 6 7 8 9 10 11
    ┌─────┬─────┬─────┐
  0 │ U0,0│ U1,0│ U2,0│ 0
  1 │     │     │     │ 1
  2 │     │     │     │ 2
  3 │     │     │     │ 3
W 4 ├─────┼─────┼─────┤ 4 E
e 5 │ U0,1│ U1,1│ U2,1│ 5 a
s 6 │     │     │     │ 6 s
t 7 │     │     │     │ 7 t
    └─────┴─────┴─────┘
      0 1 2 3 4 5 6 7 8 9 10 11
          South (S)
```

**Ports per side:**
- North: 0-11 (12 ports)
- South: 0-11 (12 ports)
- East: 0-7 (8 ports)
- West: 0-7 (8 ports)

---

## 2×3 Foundation (34×54 grid)

**Dimensions:** 2 units wide × 3 units tall
**Total Ports:** 40 (N:8 + S:8 + E:12 + W:12)

```
      North (N)
      0 1 2 3 4 5 6 7
    ┌───────┬───────┐
  0 │ U0,0  │ U1,0  │ 0
  1 │       │       │ 1
  2 │       │       │ 2
  3 │       │       │ 3
W 4 ├───────┼───────┤ 4
e 5 │ U0,1  │ U1,1  │ 5
s 6 │       │       │ 6
t 7 │       │       │ 7
  8 ├───────┼───────┤ 8 E
  9 │ U0,2  │ U1,2  │ 9 a
 10 │       │       │ 10 s
 11 │       │       │ 11 t
    └───────┴───────┘
      0 1 2 3 4 5 6 7
      South (S)
```

**Ports per side:**
- North: 0-7 (8 ports)
- South: 0-7 (8 ports)
- East: 0-11 (12 ports)
- West: 0-11 (12 ports)

---

## 3×3 Foundation (54×54 grid)

**Dimensions:** 3 units × 3 units
**Total Ports:** 48 (4 sides × 12 ports/side)

```
            North (N)
      0 1 2 3 4 5 6 7 8 9 10 11
    ┌─────┬─────┬─────┐
  0 │U0,0 │U1,0 │U2,0 │ 0
  1 │     │     │     │ 1
  2 │     │     │     │ 2
  3 │     │     │     │ 3
W 4 ├─────┼─────┼─────┤ 4
e 5 │U0,1 │U1,1 │U2,1 │ 5
s 6 │     │     │     │ 6
t 7 │     │     │     │ 7
  8 ├─────┼─────┼─────┤ 8 E
  9 │U0,2 │U1,2 │U2,2 │ 9 a
 10 │     │     │     │ 10 s
 11 │     │     │     │ 11 t
    └─────┴─────┴─────┘
      0 1 2 3 4 5 6 7 8 9 10 11
            South (S)
```

**Ports per side:** 0-11 on all sides (12 each)

---

## T-Shape Foundation (3×2, irregular)

**Layout:**
```
1,1,1  (top row: 3 units)
0,1,0  (bottom: only middle)
```

**Visual:**
```
        North (N)
      0 1 2 3 4 5 6 7 8 9 10 11
    ┌─────┬─────┬─────┐
  0 │ U0  │ U1  │ U2  │ 0
  1 │     │     │     │ 1
  2 │     │     │     │ 2
W 3 │     │  ├──┤     │ 3 E
e 4 └─────┘  │  └─────┘ 4 a
s 5          │U3       5 s
t 6          │         6 t
  7          └─────    7
            4 5 6 7
           South (S)
```

**Ports per side:**
- North: 0-11 (12 ports - across all 3 top units)
- South: 4-7 (4 ports - only on middle bottom unit)
- East: 0-3 (4 ports - rightmost unit top portion only)
- West: 0-3 (4 ports - leftmost unit top portion only)

---

## L-Shape Foundation (2×2, irregular)

**Layout:**
```
1,1  (top row: both units)
1,0  (bottom: left only)
```

**Visual:**
```
      North (N)
      0 1 2 3 4 5 6 7
    ┌───────┬───────┐
  0 │ U0,0  │ U1,0  │ 0
  1 │       │       │ 1
  2 │       │       │ 2
  3 │       │       │ 3
W 4 ├───────┘       ├ 4 E
e 5 │ U0,1          │ 5 a
s 6 │               │ 6 s
t 7 │               │ 7 t
    └───────
      0 1 2 3
      South (S)
```

**Ports per side:**
- North: 0-7 (8 ports)
- South: 0-3 (4 ports - left unit only)
- East: 0-3 (4 ports - right unit top only)
- West: 0-7 (8 ports - both left units)

---

## L4-Shape Foundation (3×2, irregular)

**Layout:**
```
1,0,0  (top: left only)
1,1,1  (bottom: all 3)
```

**Visual:**
```
      North (N)
      0 1 2 3
    ┌─────┐
  0 │ U0  │       0
  1 │     │       1
  2 │     │       2
  3 │     │       3
W 4 ├─────┼─────┬─────┐ 4
e 5 │ U1  │ U2  │ U3  │ 5
s 6 │     │     │     │ 6 E
t 7 │     │     │     │ 7 a
    └─────┴─────┴─────┘   s
      0 1 2 3 4 5 6 7 8 9 10 11 t
            South (S)
```

**Ports per side:**
- North: 0-3 (4 ports - top left unit)
- South: 0-11 (12 ports - all 3 bottom units)
- East: 4-7 (4 ports - rightmost bottom unit)
- West: 0-7 (8 ports - both left units)

---

## S4-Shape Foundation (3×2, irregular)

**Layout:**
```
1,1,0  (top: left and middle)
0,1,1  (bottom: middle and right)
```

**Visual:**
```
        North (N)
      0 1 2 3 4 5 6 7
    ┌─────┬─────┐
  0 │ U0  │ U1  │     0
  1 │     │     │     1
  2 │     │     │     2
  3 │     │  ├──┼─────┐ 3
W 4 └─────┘  │  │ U2  │ 4 E
e 5          │U3│     │ 5 a
s 6          │  │     │ 6 s
t 7          └──┴─────┘ 7 t
            4 5 6 7 8 9 10 11
              South (S)
```

**Ports per side:**
- North: 0-7 (8 ports - top two units)
- South: 4-11 (8 ports - bottom two units)
- East: 4-7 (4 ports - right bottom unit)
- West: 0-3 (4 ports - left top unit)

---

## Cross (+) Foundation (3×3, irregular)

**Layout:**
```
0,1,0  (top: middle only)
1,1,1  (middle: all 3)
0,1,0  (bottom: middle only)
```

**Visual:**
```
          North (N)
          4 5 6 7
          ┌─────┐
        0 │ U1  │ 0
        1 │     │ 1
        2 │     │ 2
        3 │     │ 3
    ┌─────┼─────┼─────┐
  4 │ U0  │ U2  │ U4  │ 4
W 5 │     │     │     │ 5 E
e 6 │     │     │     │ 6 a
s 7 │     │     │     │ 7 s
t   └─────┼─────┼─────┘   t
      8   │ U3  │   8
      9   │     │   9
     10   │     │   10
     11   └─────┘   11
          4 5 6 7
          South (S)
```

**Ports per side:**
- North: 4-7 (4 ports - top middle unit)
- South: 4-7 (4 ports - bottom middle unit)
- East: 4-7 (4 ports - right middle unit)
- West: 4-7 (4 ports - left middle unit)

---

## Quick Reference Table

| Foundation | Total Ports | North | South | East | West | Grid Size |
|------------|-------------|-------|-------|------|------|-----------|
| 1×1        | 16          | 0-3   | 0-3   | 0-3  | 0-3  | 14×14     |
| 2×1        | 24          | 0-7   | 0-7   | 0-3  | 0-3  | 34×14     |
| 1×2        | 24          | 0-3   | 0-3   | 0-7  | 0-7  | 14×34     |
| 2×2        | 32          | 0-7   | 0-7   | 0-7  | 0-7  | 34×34     |
| 3×1        | 32          | 0-11  | 0-11  | 0-3  | 0-3  | 54×14     |
| 3×2        | 40          | 0-11  | 0-11  | 0-7  | 0-7  | 54×34     |
| 2×3        | 40          | 0-7   | 0-7   | 0-11 | 0-11 | 34×54     |
| 3×3        | 48          | 0-11  | 0-11  | 0-11 | 0-11 | 54×54     |
| T          | 32          | 0-11  | 4-7   | 0-3  | 0-3  | 54×34     |
| L          | 24          | 0-7   | 0-3   | 0-3  | 0-7  | 34×34     |
| L4         | 32          | 0-3   | 0-11  | 4-7  | 0-7  | 54×34     |
| S4         | 32          | 0-7   | 4-11  | 4-7  | 0-3  | 54×34     |
| Cross      | 16          | 4-7   | 4-7   | 4-7  | 4-7  | 54×54     |

---

## Port Specification Format

```
Side,Position,Floor,ShapeCode
```

**Examples:**
```
W,0,0,CuCuCuCu     # West side, port 0, floor 0, full circle
N,5,1,Ru------     # North side, port 5, floor 1, NE corner
E,11,2,CrCgCbCy    # East side, port 11, floor 2, 4 colors
S,7,3,RuRuRuRu:CuCuCuCu  # South port 7, floor 3, 2-layer stack
```

---

## Pro Tips

1. **Port numbering starts at 0**, not 1
2. **Larger foundations = more ports** for high-throughput operations
3. **Use multiple floors** (0-3) to route parallel streams without collisions
4. **Irregular shapes** have fewer ports but save space
5. **Each port is a 1×4 notch** in the actual game
