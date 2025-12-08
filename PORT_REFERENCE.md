# Port Reference Guide - Visual Port Numbering

This guide shows the exact port positions for each foundation type.

## Port Position Numbering

Ports are numbered **0, 1, 2, 3...** along each side, starting from the **bottom-left corner** of each side and going **up/right**.

## 1x1 Foundation (14×14 grid)

```
        NORTH (N)
      0   1   2   3
    ┌───┬───┬───┬───┐
  0 │               │ 0
W 1 │               │ 1 E
E 2 │    1x1        │ 2 A
S 3 │               │ 3 S
T   └───┴───┴───┴───┘   T
      0   1   2   3
        SOUTH (S)
```

**Port Specifications:**
- **W,0,0,Shape** = West side, position 0 (bottom), floor 0
- **N,2,0,Shape** = North side, position 2 (from left), floor 0
- **E,3,0,Shape** = East side, position 3 (top), floor 0
- **S,1,0,Shape** = South side, position 1 (from left), floor 0

---

## 2x2 Foundation (34×34 grid)

```
             NORTH (N)
      0   1   2   3   4   5   6   7
    ┌───┬───┬───┬───┬───┬───┬───┬───┐
  0 │                               │ 0
W 1 │                               │ 1
E 2 │                               │ 2
S 3 │            2x2                │ 3 E
T 4 │                               │ 4 A
  5 │                               │ 5 S
  6 │                               │ 6 T
  7 │                               │ 7
    └───┴───┴───┴───┴───┴───┴───┴───┘
      0   1   2   3   4   5   6   7
             SOUTH (S)
```

**Port Specifications:**
- **W,0,0,Shape** = West side, position 0 (bottom), floor 0
- **W,4,0,Shape** = West side, position 4 (middle), floor 0
- **N,7,0,Shape** = North side, position 7 (rightmost), floor 0
- **E,0,1,Shape** = East side, position 0 (bottom), floor 1
- **S,3,2,Shape** = South side, position 3, floor 2

**Ports per side:** 8 (positions 0-7)

---

## 3x3 Foundation (54×54 grid)

```
                    NORTH (N)
      0   1   2   3   4   5   6   7   8   9  10  11
    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  0 │                                               │ 0
W 1 │                                               │ 1
E 2 │                                               │ 2
S 3 │                                               │ 3
T 4 │                   3x3                         │ 4 E
  5 │                                               │ 5 A
  6 │                                               │ 6 S
  7 │                                               │ 7 T
  8 │                                               │ 8
  9 │                                               │ 9
 10 │                                               │ 10
 11 │                                               │ 11
    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
      0   1   2   3   4   5   6   7   8   9  10  11
                    SOUTH (S)
```

**Ports per side:** 12 (positions 0-11)

---

## 2x1 Foundation (34×14 grid)

```
             NORTH (N)
      0   1   2   3   4   5   6   7
    ┌───┬───┬───┬───┬───┬───┬───┬───┐
  0 │                               │ 0
W 1 │                               │ 1 E
E 2 │            2x1                │ 2 A
S 3 │                               │ 3 S
T   └───┴───┴───┴───┴───┴───┴───┴───┘   T
      0   1   2   3   4   5   6   7
             SOUTH (S)
```

**Ports:**
- **North/South:** 8 ports (positions 0-7)
- **East/West:** 4 ports (positions 0-3)

---

## 1x2 Foundation (14×34 grid)

```
        NORTH (N)
      0   1   2   3
    ┌───┬───┬───┬───┐
  0 │               │ 0
  1 │               │ 1
  2 │               │ 2
W 3 │               │ 3
E 4 │     1x2       │ 4 E
S 5 │               │ 5 A
T 6 │               │ 6 S
  7 │               │ 7 T
    └───┴───┴───┴───┘
      0   1   2   3
        SOUTH (S)
```

**Ports:**
- **North/South:** 4 ports (positions 0-3)
- **East/West:** 8 ports (positions 0-7)

---

## Quick Reference Table

| Foundation | North/South Ports | East/West Ports | Grid Size |
|------------|------------------|-----------------|-----------|
| 1x1        | 4 (0-3)         | 4 (0-3)         | 14×14     |
| 2x1        | 8 (0-7)         | 4 (0-3)         | 34×14     |
| 1x2        | 4 (0-3)         | 8 (0-7)         | 14×34     |
| 2x2        | 8 (0-7)         | 8 (0-7)         | 34×34     |
| 3x2        | 12 (0-11)       | 8 (0-7)         | 54×34     |
| 2x3        | 8 (0-7)         | 12 (0-11)       | 34×54     |
| 3x3        | 12 (0-11)       | 12 (0-11)       | 54×54     |

---

## Examples with Visual Reference

### Example 1: Simple 1x1 Corner Splitter

**Input:**
```
W,0,0,CuCuCuCu    <- West side, bottom port, full circle
```

**Outputs:**
```
E,0,0,Cu------    <- East side, bottom port, NE corner only
E,1,0,--Cu----    <- East side, port 1, NW corner only
E,2,0,----Cu--    <- East side, port 2, SW corner only
E,3,0,------Cu    <- East side, top port, SE corner only
```

**Visual:**
```
        NORTH
      0   1   2   3
    ┌───┬───┬───┬───┐
  0 │               ├─→ E,0,0 (Cu------)
W 1 │   Cutters     ├─→ E,1,0 (--Cu----)
E 2 │   Route       ├─→ E,2,0 (----Cu--)
S 3 │   Here        ├─→ E,3,0 (------Cu)
T   └───┴───┴───┴───┘
 ↑    0   1   2   3
W,0,0    SOUTH
```

---

## Floor Numbers

- **Floor 0**: Bottom/ground level
- **Floor 1**: First elevated level
- **Floor 2**: Second elevated level
- **Floor 3**: Top level

Belts can route on different floors to avoid conflicts!

---

## Pro Tips

1. **Port positions always start at 0** (not 1)
2. **Positions increase left-to-right** for N/S sides
3. **Positions increase bottom-to-top** for E/W sides
4. **Use multiple floors** to route more connections without collisions
5. **Larger foundations = more ports** (1x1 has 4, 3x3 has 12 per side)
