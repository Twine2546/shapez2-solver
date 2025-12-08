# Shapez 2 CP-SAT Solver - Foundation Examples

This guide shows how to use the CP-SAT solver GUI for different foundation types and common use cases.

## Shape Code Format

### Basic Structure
```
ShapeCode: 8 characters = 4 quadrants × 2 characters each
Quadrants: NE, NW, SW, SE (clockwise from top-right)
Each quadrant: ShapeType + Color
Layers: Separated by : for stacked shapes
```

### Shape Types
- `C` = Circle
- `R` = Square (Rectangle)
- `S` = Star
- `W` = Diamond (Windmill)
- `c` = Crystal
- `P` = Pin (no color needed)
- `-` = Empty

### Colors
- `u` = Uncolored
- `r` = Red
- `g` = Green
- `b` = Blue
- `c` = Cyan
- `m` = Magenta
- `y` = Yellow
- `w` = White

### Example Shape Codes
```
CuCuCuCu          Full circle (all 4 quadrants)
Cu------          NE corner only
--Cu----          NW corner only
----Cu--          SW corner only
------Cu          SE corner only
CrCgCbCy          Circle with 4 different colors
RrRrRrRr:CuCuCuCu Two layers: red squares (bottom) + circles (top)
SuSuSuSu          Full star
WyWyWyWy          Yellow diamond (all quadrants)
```

## Port Specification Format

```
Side,Position,Floor,ShapeCode
```

- **Side**: `N` (North), `E` (East), `S` (South), `W` (West)
- **Position**: Port number along the side
- **Floor**: 0-3 (4 floors available)
- **ShapeCode**: The shape at this port

## Foundation Types and Port Layouts

### 1x1 Foundation (14×14 internal grid)
- **Ports per side**: 4 (positions 0-3)
- **Use case**: Simple transformations, basic splitting

**Example: Corner Splitter**
```
Input:
W,0,0,CuCuCuCu

Outputs:
E,0,0,Cu------
E,1,0,--Cu----
E,2,0,----Cu--
E,3,0,------Cu
```
This takes a full circle and splits it into 4 individual corners.

---

### 2x1 Foundation (34×14 internal grid)
- **North/South sides**: 8 ports (positions 0-7)
- **East/West sides**: 4 ports (positions 0-3)
- **Use case**: Horizontal splitting, linear processing

**Example: Horizontal Multi-Splitter**
```
Input:
W,0,0,RuRuRuRu
W,1,0,RuRuRuRu

Outputs:
E,0,0,Ru------
E,1,0,--Ru----
E,2,0,----Ru--
E,3,0,------Ru
```

---

### 1x2 Foundation (14×34 internal grid)
- **North/South sides**: 4 ports (positions 0-3)
- **East/West sides**: 8 ports (positions 0-7)
- **Use case**: Vertical splitting, stacking operations

**Example: Vertical Processing**
```
Input:
W,0,0,SuSuSuSu

Outputs:
E,0,0,Su------
E,1,0,--Su----
E,2,0,----Su--
E,3,0,------Su
E,4,0,Su------
E,5,0,--Su----
```

---

### 2x2 Foundation (34×34 internal grid)
- **Ports per side**: 8 (positions 0-7)
- **Use case**: Complex splitting, multi-floor operations, standard use

**Example 1: Corner Splitter on Multiple Floors**
```
Input:
W,0,0,CuCuCuCu

Outputs:
E,0,0,Cu------
E,1,0,--Cu----
E,2,1,----Cu--
E,3,1,------Cu
```

**Example 2: Multi-Input Processing**
```
Inputs:
W,0,0,RuRuRuRu
W,1,0,CuCuCuCu

Outputs:
E,0,0,Ru------
E,1,0,--Ru----
E,2,0,Cu------
E,3,0,--Cu----
```

**Example 3: Color Separation**
```
Input:
W,0,0,CrCgCbCy

Outputs:
E,0,0,Cr------
E,1,0,--Cg----
E,2,0,----Cb--
E,3,0,------Cy
```

---

### 3x2 Foundation (54×34 internal grid)
- **North/South sides**: 12 ports (positions 0-11)
- **East/West sides**: 8 ports (positions 0-7)
- **Use case**: Large horizontal splitting operations

**Example: High-Throughput Splitter**
```
Input:
W,0,0,CuCuCuCu

Outputs:
E,0,0,Cu------
E,1,0,--Cu----
E,2,0,----Cu--
E,3,0,------Cu
E,4,0,Cu------
E,5,0,--Cu----
E,6,0,----Cu--
E,7,0,------Cu
```

---

### 2x3 Foundation (34×54 internal grid)
- **North/South sides**: 8 ports (positions 0-7)
- **East/West sides**: 12 ports (positions 0-11)
- **Use case**: Vertical high-throughput operations

**Example: Vertical Multi-Splitter**
```
Inputs:
W,0,0,SuSuSuSu
W,1,0,SuSuSuSu

Outputs:
E,0,0,Su------
E,1,0,--Su----
E,2,0,----Su--
E,3,0,------Su
E,4,0,Su------
E,5,0,--Su----
```

---

### 3x3 Foundation (54×54 internal grid)
- **Ports per side**: 12 (positions 0-11)
- **Use case**: Maximum complexity, highest throughput

**Example 1: Maximum Splitting (1→16)**
```
Input:
W,0,0,CbCbCbCb

Outputs:
E,0,0,CbCbCbCb
E,1,0,CbCbCbCb
E,2,0,CbCbCbCb
E,3,0,CbCbCbCb
E,4,0,CbCbCbCb
E,5,0,CbCbCbCb
E,6,0,CbCbCbCb
E,7,0,CbCbCbCb
E,8,0,CbCbCbCb
E,9,0,CbCbCbCb
E,10,0,CbCbCbCb
E,11,0,CbCbCbCb
```

**Example 2: Multi-Floor Processing (12→48)**
```
Inputs:
W,0,0,RuCuSuWu
W,1,0,RuCuSuWu
W,2,0,RuCuSuWu
W,3,0,RuCuSuWu
W,0,1,RuCuSuWu
W,1,1,RuCuSuWu
W,2,1,RuCuSuWu
W,3,1,RuCuSuWu
W,0,2,RuCuSuWu
W,1,2,RuCuSuWu
W,2,2,RuCuSuWu
W,3,2,RuCuSuWu

Outputs:
(48 outputs - 4 corners × 12 inputs)
E,0,0,Ru------
E,1,0,--Cu----
E,2,0,----Su--
E,3,0,------Wu
... (and so on for all outputs)
```

---

## Common Use Cases

### 1. Simple Corner Splitter
**Goal**: Split a full shape into 4 individual corners

**Foundation**: 2x2
```
Input:  W,0,0,CuCuCuCu
Output: E,0,0,Cu------
        E,1,0,--Cu----
        E,2,0,----Cu--
        E,3,0,------Cu
```
**Expected**: 3 cutters, 11.25 items/min per output

---

### 2. Pure Splitting (No Transformation)
**Goal**: Duplicate shape to multiple outputs (no cutting needed)

**Foundation**: 2x2
```
Input:  W,0,0,RuRuRuRu
Output: E,0,0,RuRuRuRu
        E,1,0,RuRuRuRu
        E,2,0,RuRuRuRu
        E,3,0,RuRuRuRu
```
**Expected**: 3 splitters, 45 items/min per output (4× faster than cutters!)

---

### 3. Color Extraction
**Goal**: Extract specific colors from a multi-colored shape

**Foundation**: 2x2
```
Input:  W,0,0,CrCgCbCy
Output: E,0,0,Cr------  (red only)
        E,1,0,--Cg----  (green only)
        E,2,0,----Cb--  (blue only)
        E,3,0,------Cy  (yellow only)
```

---

### 4. Multi-Layer Stacking
**Goal**: Work with 2-layer stacked shapes

**Foundation**: 2x2
```
Input:  W,0,0,RuRuRuRu:CuCuCuCu
Output: E,0,0,RuRuRuRu:CuCuCuCu
        E,1,0,RuRuRuRu:CuCuCuCu
```

---

### 5. Multi-Floor Parallel Processing
**Goal**: Process multiple inputs in parallel on different floors

**Foundation**: 2x2
```
Inputs: W,0,0,SuSuSuSu
        W,0,1,CuCuCuCu
        W,0,2,RuRuRuRu

Outputs: E,0,0,Su------
         E,1,0,--Su----
         E,0,1,Cu------
         E,1,1,--Cu----
         E,0,2,Ru------
         E,1,2,--Ru----
```

---

## Throughput Optimization

The CP-SAT solver automatically optimizes for maximum throughput:

### Machine Speeds (Tier 5 - Fully Upgraded)
- **Belt**: 180 items/min
- **Splitter**: 180 items/min (NO processing bottleneck)
- **Cutter**: 45 items/min (4× slower than splitters!)

### Smart Machine Selection
1. **Pure Splitting** (no shape transformation): Uses **splitters** for 180 items/min throughput
2. **Shape Transformation** (cutting required): Uses **cutters** at 45 items/min

### Example Comparison
**Scenario**: 1 input → 4 outputs

**With Shape Transformation** (corner splitting):
- Uses 3 cutters
- Throughput: 11.25 items/min per output (45/4)

**Without Transformation** (pure duplication):
- Uses 3 splitters
- Throughput: 45 items/min per output (180/4)
- **4× faster!**

---

## Tips for Best Results

1. **Choose the Right Foundation**:
   - Start with 2x2 for most tasks
   - Use 3x3 for complex splitting (>8 outputs)
   - Use smaller foundations (1x1, 2x1) for simple operations

2. **Leverage Multiple Floors**:
   - Process independent input streams in parallel
   - Avoid routing conflicts by using different floors

3. **Maximize Throughput**:
   - Prefer splitters over cutters when possible
   - Use pure splitting (no transformation) for 4× speed boost

4. **Port Numbering**:
   - Positions are numbered along each side
   - 1x1: 0-3 per side
   - 2x2: 0-7 per side
   - 3x3: 0-11 per side

5. **Testing**:
   - Start simple (1→2 or 1→4)
   - Gradually increase complexity
   - Use "View Layout" to visualize the solution

---

## Troubleshooting

### "No solution found"
- Try a larger foundation
- Reduce the number of outputs
- Check shape codes are valid (8 characters each)
- Ensure port positions are within range for your foundation

### "Routing failed"
- Foundation may be too crowded
- Try using different floors
- Consider splitting the task into multiple foundations

### Low Throughput
- Check if you're using cutters when splitters would work
- Reduce the split ratio (fewer outputs per input)
- Use multiple input sources for parallel processing

---

## Quick Start Examples

### Beginner: 1→2 Corner Split
```
Foundation: 2x2
Input:  W,0,0,CuCuCuCu
Output: E,0,0,Cu------
        E,1,0,--Cu----
```

### Intermediate: 1→4 Full Corner Split
```
Foundation: 2x2
Input:  W,0,0,RuRuRuRu
Output: E,0,0,Ru------
        E,1,0,--Ru----
        E,2,0,----Ru--
        E,3,0,------Ru
```

### Advanced: 4→16 Multi-Floor
```
Foundation: 3x3
Inputs: W,0,0,CrCgCbCy
        W,1,0,CrCgCbCy
        W,2,0,CrCgCbCy
        W,3,0,CrCgCbCy
Outputs: (16 total - 4 corners each)
        E,0,0,Cr------
        E,1,0,--Cg----
        E,2,0,----Cb--
        E,3,0,------Cy
        ... (repeat pattern for other 3 inputs)
```

---

## GUI Usage

1. **Select Foundation**: Choose foundation type from dropdown
2. **Enter Inputs**: One per line in format `Side,Pos,Floor,Shape`
3. **Enter Outputs**: One per line in same format
4. **Click "Solve"**: Wait for solution
5. **View Results**: Check fitness, throughput, and building counts
6. **View Layout**: Click to see visual layout
7. **Copy Blueprint**: Copy code to clipboard for import into Shapez 2

---

For more information, see the main README.md file.
