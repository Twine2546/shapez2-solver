# Shapez 2 Solver Design Document

## Problem Domain

We're solving factory layout problems for Shapez 2. Given:
- A foundation (grid of cells)
- Input ports (shapes entering from edges)
- Output ports (shapes exiting to edges)

Find a valid placement of machines and belts that transforms inputs into outputs.

---

## Core Concepts

### 1. Flow Model

Items flow through the factory as **streams**. Each stream has:
- **Shape code**: What shape is being transported (e.g., "CuCuCuCu" = 4 copper quarters)
- **Throughput**: Items per minute (max 180/min for tier 5 belt)
- **Source**: Where the stream originates (input port or machine output)
- **Destination**: Where the stream goes (output port or machine input)

### 2. Buildings

#### Belts (Transport)
| Type | Function | Throughput |
|------|----------|------------|
| Belt | Move items in direction | 180/min |
| Belt Turn | Change direction 90° | 180/min |
| Lift | Move between floors | 180/min |
| Belt Port | Teleport up to 4 cells | 180/min |
| Splitter | 1 input → 2 outputs (alternating) | 180/min total |
| Merger | 2 inputs → 1 output | 180/min total |

#### Machines (Transform)
| Type | Size | Inputs | Outputs | Throughput | Function |
|------|------|--------|---------|------------|----------|
| Cutter | 1x2 | 1 | **2** | 45/min | Split shape into left/right halves |
| Half-Cutter | 1x1 | 1 | 1 | 45/min | Destroy one half |
| Rotator | 1x1 | 1 | 1 | 90/min | Rotate shape CW/CCW/180 |
| Stacker | 1x1x2 | 2 | 1 | 30/min | Stack two shapes |
| Swapper | 2x2 | 2 | 2 | 45/min | Swap quadrants |
| Painter | 1x1 | 2 | 1 | 45/min | Apply color |

### 3. Port Positions

Each building has input/output ports at specific positions relative to its origin:

```
CUTTER (1x2, facing EAST):
  Origin at (x, y)
  Body cells: (x, y), (x, y+1)

  Input port: (x-1, y)     ← items enter from WEST
  Output 0:   (x+1, y)     ← LEFT half exits EAST at y
  Output 1:   (x+1, y+1)   ← RIGHT half exits EAST at y+1

     [IN]→[BODY]→[OUT0]
          [BODY]→[OUT1]
```

**Critical**: A machine with N outputs produces N separate streams. ALL streams must have destinations or the machine backs up and stops.

---

## Flow Graph Model

### Nodes
1. **Input Port**: Source node, produces a stream
2. **Output Port**: Sink node, consumes a stream
3. **Machine Input**: Consumes a stream
4. **Machine Output**: Produces a stream
5. **Belt Cell**: Passes through a stream (or splits/merges)

### Edges
Connections between nodes representing item flow:
- Input Port → Machine Input (or Belt)
- Machine Output → Machine Input (chaining)
- Machine Output → Output Port
- Machine Output → Belt → ... → destination

### Validation Rules

A solution is **valid** if and only if:

1. **All sources have destinations**: Every machine output port connects to something
2. **All sinks have sources**: Every machine input port receives from something
3. **Shape compatibility**: Source shape matches what destination expects
4. **Throughput balance**: No belt/machine is over capacity
5. **No physical overlap**: Buildings don't occupy same cells
6. **Port accessibility**: Ports aren't blocked by other building bodies

---

## Example: 2 Inputs → 2 Outputs with Cutting

```
Inputs:  [CuCuCuCu] x2 @ 180/min each = 360/min total
Outputs: [Cu------] x2 @ 180/min each = 360/min total
```

### Shape Analysis
- Input has 4 quarters (full shape)
- Output has 1 quarter (quarter shape)
- Need to CUT the input, then only use LEFT half

### Flow Graph
```
Input0 (CuCuCuCu, 180/min)
    │
    ▼
┌─────────┐
│ Cutter0 │ (45/min capacity)
└─────────┘
    │ Output0: [CuCu----] LEFT half (45/min)
    │ Output1: [----CuCu] RIGHT half (45/min) → TRASH or merge
    ▼
Output0 (need Cu------)

    ❌ PROBLEM: Cutter only processes 45/min but input is 180/min!
    ❌ PROBLEM: Output is [CuCu----] but we need [Cu------]!
```

### Correct Solution
Need 4 cutters per input belt for saturation:
```
Input0 (180/min) ──┬── Cutter0 (45/min) ── [CuCu----] ──┐
                   ├── Cutter1 (45/min) ── [CuCu----] ──┤
                   ├── Cutter2 (45/min) ── [CuCu----] ──┼── Merger ── Output0
                   └── Cutter3 (45/min) ── [CuCu----] ──┘

                   (+ 4 more cutters for RIGHT halves → trash or reuse)
```

---

## Solver Architecture

### Phase 1: Recipe Analysis
Given input/output shapes, determine:
1. What transformations are needed (cut, rotate, stack, etc.)
2. How many machines of each type
3. Throughput requirements and machine count for saturation

### Phase 2: Flow Graph Construction
Build the logical flow graph:
1. Create nodes for all inputs, outputs, machine ports
2. Assign which machine handles which input stream
3. Route each machine output to a destination
4. Verify all ports are connected

### Phase 3: Physical Placement
Place machines on the grid:
1. Use CP-SAT to find non-overlapping positions
2. Ensure port accessibility (ports not blocked)
3. Consider routing feasibility (machines not too far from I/O)

### Phase 4: Belt Routing
Connect the flow graph physically:
1. For each edge in flow graph, find a belt path
2. Use A* pathfinding with constraints
3. Handle belt ports for long distances
4. Verify no belt conflicts

### Phase 5: Validation
Check the complete solution:
1. Trace flow from each input to outputs
2. Verify throughput at each node
3. Check all machine outputs have destinations
4. Verify shape transformations are correct

---

## Data Structures

### FlowStream
```python
@dataclass
class FlowStream:
    shape: str           # e.g., "CuCuCuCu"
    throughput: float    # items/min
    source: FlowNode     # where it comes from
    destination: FlowNode # where it goes (None if unrouted)
```

### FlowNode
```python
@dataclass
class FlowNode:
    node_type: str       # 'input_port', 'output_port', 'machine_in', 'machine_out'
    position: Tuple[int, int, int]  # (x, y, floor)
    building: Optional[Building]     # associated building
    port_index: int      # which port (for multi-port machines)

    incoming: List[FlowStream]  # streams entering this node
    outgoing: List[FlowStream]  # streams leaving this node
```

### FlowGraph
```python
class FlowGraph:
    nodes: List[FlowNode]
    streams: List[FlowStream]

    def validate(self) -> List[str]:
        """Return list of validation errors, empty if valid."""
        errors = []

        # Check all machine outputs have destinations
        for node in self.nodes:
            if node.node_type == 'machine_out':
                if not node.outgoing:
                    errors.append(f"Machine output at {node.position} has no destination")

        # Check throughput limits
        for node in self.nodes:
            total_in = sum(s.throughput for s in node.incoming)
            if total_in > node.max_throughput:
                errors.append(f"Node at {node.position} over capacity: {total_in}/{node.max_throughput}")

        return errors
```

---

## Implementation Plan

### Step 1: Flow Graph Builder
- Input: recipe (input shapes → output shapes)
- Output: logical FlowGraph with all connections
- Handles: machine selection, throughput calculation, stream routing

### Step 2: Placement Solver
- Input: FlowGraph, foundation spec
- Output: physical positions for all machines
- Uses: CP-SAT with overlap + port constraints

### Step 3: Belt Router
- Input: placed machines, FlowGraph edges
- Output: belt paths connecting all streams
- Uses: A* with conflict avoidance

### Step 4: Solution Validator
- Input: complete solution
- Output: validation report
- Traces: actual flow through the physical layout

---

## Key Insights

1. **Flow-first design**: Build the logical flow graph BEFORE physical placement
2. **All outputs matter**: Every machine output port needs a destination
3. **Throughput math**: Calculate machine counts from throughput ratios
4. **Validate everything**: Check flow graph validity before and after routing

---

## Current Issues to Fix

1. **Cutter has 2 outputs** - routing only connects 1, other backs up
2. **No flow validation** - can't verify solution actually works
3. **Throughput ignored** - not calculating proper machine counts
4. **Recipe analysis missing** - not determining what transformations are needed

## Next Steps

1. Implement `FlowGraph` class with validation
2. Rewrite recipe analyzer to build proper flow graph
3. Modify router to connect ALL streams in flow graph
4. Add post-routing flow validation
