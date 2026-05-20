# Gravitas

A 3D N-body gravitational simulation built to explore CPU and GPU parallel programming. Particles interact through Newtonian gravity using algorithms ranging from brute-force O(n²) to Barnes-Hut O(n log n), with an OpenGL renderer that includes bloom post-processing.

## How it works

Each frame, the active solver computes a gravitational force on every particle, then integrates positions and velocities forward one timestep using leapfrog integration. A `ParticleSimulation` owns a `ParticleSystem` (the particle data), a `ParticleSolver` (the physics), and a `ParticleDrawer` (the renderer). For GPU solvers, particle data lives in an SSBO that both the compute shaders and vertex shader access directly — no CPU readback per frame.

### Brute-force solvers (v1–v4)

Every particle interacts with every other — O(n²) per frame.

- **v1** — CPU sequential. Straightforward double loop.
- **v2** — CPU parallel via OpenMP. The outer force loop is parallelized across threads.
- **v3** — GPU compute shader. One thread per particle, reads all others from the SSBO.
- **v4** — GPU tiled compute shader. Shared-memory tiling from [NVIDIA GPU Gems 3 Ch. 31](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda) loads tiles of particles into local memory, reducing global memory bandwidth by a factor of the tile size.

### Barnes-Hut solvers (v5–v7)

An octree is rebuilt every frame and used to approximate distant clusters, reducing complexity to O(n log n).

The opening criterion is `s/d < θ` (θ = 0.5): if a node's size `s` divided by its distance `d` from the query particle is below the threshold, the node's aggregate mass is used instead of recursing into its children. Tree traversal is stackless — each node stores a precomputed `next` pointer so force computation is a single linked-list walk.

- **v5** — CPU sequential. Tree built and traversed on a single thread.
- **v6** — CPU parallel via OpenMP. Force traversal parallelized across particles.
- **v7** — GPU fully parallel. Tree construction and force traversal both run on the GPU entirely via compute shaders. See `BarnesHutExplained.md` for a full breakdown of the parallel construction pipeline.

Plummer softening (ε²) is applied to all force calculations to prevent singularities when particles get very close.

### GPU Barnes-Hut pipeline (v7)

The octree is built on the GPU each frame across several compute shader passes:

1. **Bounding box** — two-pass reduction to compute the world AABB from particle positions
2. **Morton codes** — each particle's 3D position is mapped to a 30-bit Morton code
3. **Bitonic sort** — particles sorted by Morton code to improve spatial locality and reduce warp divergence
4. **Octree construction** — nodes inserted in parallel using a task-based scheme; `expandOctree`, `countParticlesPerTask`, `distributeParticles`, and `insertParticles` shaders build the tree iteratively
5. **Propagation** — aggregate mass and center-of-mass computed bottom-up via `propagateFatherOctree`
6. **Force calculation** — each particle traverses the tree with the `forceCalculateBarnesHut` shader using the stackless next-pointer traversal
7. **Integration** — `updateParticles` applies leapfrog integration to advance positions and velocities

## Requirements

- `cmake` ≥ 3.16
- C++17 compiler (Clang on macOS, GCC or MSVC on Linux/Windows)
- OpenGL 4.3+
- `xorg-dev` (Linux only)
- OpenMP (optional but recommended for v2, v6)

Dependencies (glfw, glad, glm) are fetched automatically by CMake on first build.

## Build and run

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)            # Linux
make -j$(sysctl -n hw.logicalcpu)   # macOS
./N-body
```

On Windows, replace the last two steps with:
```bash
cmake --build . --config Release
.\Release\N-body.exe
```

Default run (100 particles, galaxy disk, CPU sequential):
```bash
./N-body -v 1 -n 100 -i 2 -t 0.001 -s 0.0935
```

## Arguments

| Flag | Description |
|------|-------------|
| `-v` | Solver version (1–7) |
| `-n` | Number of particles |
| `-i` | Initialization type (1–6) |
| `-t` | Time step |
| `-s` | Squared Plummer softening |
| `-f` | Path to a particle system file |

**Versions**

| `-v` | Description | Complexity |
|------|-------------|------------|
| 1 | CPU sequential brute-force | O(n²) |
| 2 | CPU parallel brute-force (OpenMP) | O(n²) |
| 3 | GPU brute-force (compute shader) | O(n²) |
| 4 | GPU tiled brute-force (shared memory) | O(n²) |
| 5 | Barnes-Hut CPU sequential | O(n log n) |
| 6 | Barnes-Hut CPU parallel (OpenMP) | O(n log n) |
| 7 | Barnes-Hut GPU (fully parallel) | O(n log n) |

> v5–v7 can become unstable with very large particle counts or near-collisions. For stable simulations use v1–v4.

**Initializations**

| `-i` | Shape |
|------|-------|
| 1 | Cube (volume) |
| 2 | Galaxy disk |
| 3 | Lagrange triangle (3 bodies) |
| 4 | Sphere surface |
| 5 | Ball (solid sphere) |
| 6 | Cube surface |

## Controls

| Input | Action |
|-------|--------|
| Esc | Close |
| Space | Pause / resume |
| Scroll | Zoom |
| Click + drag | Rotate camera |
| B | Toggle bloom |
| I / D | Increase / decrease bloom intensity |
| Q | Toggle point size |

## Program structure

```
main.cpp
├── RenderLoop            — GLFW window loop, dispatches update/draw each frame
├── ParticleSimulation    — owns the system, solver, and drawer; handles CPU/GPU sync
│   ├── ParticleSystem    — particle data (position, velocity, mass); manages the SSBO
│   ├── ParticleSolver    — abstract base; one concrete class per version (v1–v7)
│   │   ├── v1  ParticleSolverCPUSequential
│   │   ├── v2  ParticleSolverCPUParallel
│   │   ├── v3  ParticleSolverGPU
│   │   ├── v4  ParticleSolverGPUTiled         (forceCalculationOptimized.glsl)
│   │   ├── v5  ParticleSolverBHutCPUSeq
│   │   ├── v6  ParticleSolverBHutCPUParallel
│   │   └── v7  ParticleSolverBHutGPU
│   │       └── ParallelOctreeGPU             — manages all BH compute shader passes
│   └── ParticleDrawer    — vertex+fragment shader render; Bloom post-processing
├── Camera                — arcball camera, projection/view matrices
├── WindowInputManager    — GLFW keyboard/mouse callbacks
└── ArgumentsParser       — CLI flag parsing
```

To add a new solver: implement the `ParticleSolver` interface and wire it into `main.cpp` and `enums.h`.  
To add a new initialization: implement `ParticleSystemInitializer` and wire it into `main.cpp` and `enums.h`.

## Particle system file format

```
Particle System with 3 particles:
Particle ID: 0
Position: (5.8337, 4.6008, 3.35599)
Velocity: (-1.15428, 1.8317, 0)
Acceleration: (0, 0, 0)
Mass: 0.625
...
```

World dimensions are (5, 5, 5). Pass the file path with `-f path/to/file`.
