#ifndef N_BODY_ENUMS_H
#define N_BODY_ENUMS_H

// How the initial particle positions, velocities, and masses are generated
enum InitializationType{
    CUBE = 1,        // particles scattered randomly inside a cube
    GALAXY = 2,      // rotating disk (galaxy approximation)
    LAGRANGE = 3,    // 3-body equilateral triangle, zero initial velocity
    SPHERE = 4,      // particles on the surface of a sphere
    BALL = 5,        // particles distributed throughout a solid ball
    CUBE_SURFACE = 6,// particles on the surface of a cube
    SYSTEM_FILE = 7  // load from a particle system file (-f flag)
};

// Solver algorithm and execution target
enum Version {
    PP_CPU_SEQUENTIAL = 1,   // O(n^2) brute-force, single CPU thread
    PP_CPU_PARALLEL = 2,     // O(n^2) brute-force, OpenMP multi-thread
    PP_GPU_PARALLEL = 3,     // O(n^2) brute-force, GPU compute shader
    PP_GPU_OPTIMIZED = 4,    // O(n^2) tiled GPU (shared-memory, GPU Gems 3 ch. 31)
    BARNES_HUT_CPU_SEQ = 5,  // O(n log n) Barnes-Hut, single CPU thread
    BARNES_HUT_CPU_PARALLEL = 6, // O(n log n) Barnes-Hut, OpenMP multi-thread
    BARNES_HUT_GPU_PARALLEL = 7  // O(n log n) Barnes-Hut, fully on GPU
};

#endif //N_BODY_ENUMS_H
