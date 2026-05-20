# Barnes-Hut GPU Implementation

This document explains how the fully GPU Barnes-Hut solver (v7) builds and traverses the octree each frame entirely on the GPU using OpenGL compute shaders.

## Overview

The CPU sequential version (v5) rebuilds the tree on the host every frame — straightforward but single-threaded. The GPU version parallelizes every phase: bounding box computation, tree construction, mass propagation, pruning, and force evaluation. The tree layout uses a flat node array with `firstChild` and `next` pointers packed into the node struct (see `Node.h`), which maps cleanly to GPU buffer objects.

## Phase 1 — Bounding box

A parallel reduction over all particle positions computes the axis-aligned bounding box that becomes the root node's spatial extent. This runs in O(log n) with n/2 threads halving the problem each step.

## Phase 2 — Tree expansion

Rather than inserting particles one by one (which would require atomic locks), the GPU pre-builds the upper levels of the tree structure top-down before any particles are assigned. Starting from the root, each level spawns 8 child nodes per existing node in parallel. This continues down to a fixed depth, producing a skeleton of empty nodes that defines the coarse spatial partitioning. No particle data is touched in this phase.

## Phase 3 — Particle assignment

Each particle is mapped to the leaf-level subtree root it falls into. This is done in three parallel passes:

1. **Count** — count how many particles land in each subtree.
2. **Prefix sum** — scan the counts to get the write offset for each subtree's particle list.
3. **Scatter** — write each particle index into its subtree's slot in a shared index array.

The result is a compactly grouped index array: all particles for subtree 0 first, then subtree 1, and so on.

## Phase 4 — Parallel insertion

Each subtree from Phase 2 becomes an independent insertion task. Because the tasks operate on non-overlapping regions of the node array, they run concurrently without any synchronization between them. Within a task, particles are inserted one by one into that subtree's nodes, subdividing further as needed (identical to the sequential algorithm). Leaf nodes store the particle's mass and position as the node's center of mass.

## Phase 5 — Mass propagation

Once all particles are inserted, internal node masses and centers of mass are computed bottom-up. A kernel sweeps level by level from the deepest nodes toward the root. At each level, every internal node sums its children's masses and computes a weighted average position. The parent array built during insertion tracks which nodes need updating so no tree traversal is required.

## Phase 6 — Pruning

Subtrees that received no particles during insertion are logically removed by setting their `firstChild` to -1 and updating `next` pointers to skip them. This is done top-down, level by level, in parallel. Pruning is what makes the force traversal efficient: the stackless walk skips entire empty branches in a single jump.

## Phase 7 — Force evaluation

One GPU thread per particle walks the pruned tree starting from the root. At each node the thread applies the Barnes-Hut opening criterion:

```
s² < θ² · d²
```

where `s` is the longest side of the node's bounding box and `d` is the distance to the particle. If the criterion is satisfied (or the node is a leaf), the node's aggregate mass is used to compute the force contribution and the thread jumps to `node.next` — a precomputed pointer that skips the entire subtree. Otherwise the thread descends into `node.firstChild`. No stack or recursion is needed.

As an additional optimization, the first 1024 nodes of the array (the upper tree levels visited by every thread) are loaded into shared memory once per work-group, so those reads hit the fast on-chip cache rather than global memory.
