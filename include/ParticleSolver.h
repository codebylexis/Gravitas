
#ifndef N_BODY_PARTICLESOLVER_H
#define N_BODY_PARTICLESOLVER_H
#include "ParticleSystem.h"

// Abstract base class for all physics solvers.
//
// Concrete subclasses implement the force calculation and time integration
// for a specific algorithm (brute-force / Barnes-Hut) and execution target
// (CPU sequential, CPU parallel, GPU).
//
// ParticleSimulation calls updateParticlePositions() every frame.
// usesGPU() controls whether CPU↔GPU sync fences are inserted (see ParticleSimulation.h).
class ParticleSolver {
public:
    virtual ~ParticleSolver() = default;
    ParticleSolver() = default;
    virtual void updateParticlePositions(ParticleSystem *particles) = 0;
    virtual float getSquaredSoftening() = 0;
    virtual bool usesGPU() = 0;
};


#endif //N_BODY_PARTICLESOLVER_H
