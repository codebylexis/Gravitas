
#ifndef PARTICLESIMULATION_H
#define PARTICLESIMULATION_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include "ParticleSystem.h"
#include "ParticleSystemCubeInitializer.h"
#include "ParticleSystemGalaxyInitializer.h"
#include "ParticleSolver.h"
#include "ParticleDrawer.h"
#include "OpenGLBuffer.h"
#include <memory>

// Owns the particle system, solver, and drawer for one simulation run.
//
// Each frame RenderLoop calls update() then draw(dt):
//   update() — advances physics by one time step
//   draw()   — renders particles with optional bloom post-processing
//
// For CPU solvers the positions live in host memory, so we use a GLsync fence
// to prevent the GPU from reading the SSBO while the CPU is writing it and
// vice-versa.  GPU solvers write directly into the SSBO, so no fence is needed
// on the host side — the driver serialises shader dispatches internally.
class ParticleSimulation {
public:
    virtual void update();
    virtual void draw(float dt);
    ParticleDrawer* getParticleDrawer() const {
        return this->particleDrawer.get();
    }
    ParticleSimulation(
        std::unique_ptr<ParticleSystemInitializer> particleSystemInitializer,
        std::unique_ptr<ParticleSolver> particleSysSolver,
        glm::vec3 worldDim,
        glm::vec2 windowDim
    );
    ~ParticleSimulation();

protected:
    std::unique_ptr<ParticleSolver> particleSolver;
    std::unique_ptr<ParticleDrawer> particleDrawer;
    std::unique_ptr<ParticleSystem> particleSystem;
    GLsync gSync = nullptr; // CPU↔GPU synchronisation fence (CPU solvers only)

    // Insert a fence after uploading CPU-side particle data to the SSBO
    void lockParticlesBuffer();
    // Block until the GPU has finished reading the SSBO from the previous frame
    void waitParticlesBuffer();
};
#endif // PARTICLESIMULATION_H
