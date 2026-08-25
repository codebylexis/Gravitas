// Barnes-Hut solver running entirely on the GPU.
// The octree is built by ParallelOctreeGPU via compute shaders, then two dispatches
// per frame do the work: one walks the tree to accumulate forces, one integrates
// positions. Particle data stays in SSBOs the whole time — nothing is read back
// to the host, so the renderer draws straight from the same buffers.

#include "ParticleSolverBHutGPU.h"

#include <chrono>
#include <iostream>
#include <glm/gtx/norm.hpp>
#include <Octree.h>
#include <ComputeShader.h>

ParticleSolverBHutGPU::ParticleSolverBHutGPU(float stepSize, float squaredSoft, int n, std::string &positionCalculatorPath, std::string &forceCalculatorPath): ParticleSolver() {
    this->squaredSoftening = squaredSoft;
    this->timeStep = stepSize;
    this->G = 1.0f;
    this->octree = std::make_unique<ParallelOctreeGPU>(n);

    this->positionCalculator = std::make_unique<ComputeShader>(positionCalculatorPath);
    this->positionCalculator->use();
    this->positionCalculator->setFloat("deltaTime", stepSize);

    this->forceCalculator = std::make_unique<ComputeShader>(forceCalculatorPath);
    this->forceCalculator->use();
    this->forceCalculator->setFloat("squaredSoftening", squaredSoft);
}

void ParticleSolverBHutGPU::updateParticlePositions(ParticleSystem *particles){
    // Recompute the bounding box and clear the task buffers
    this->octree->reset(particles);

    // TODO: Morton-order sorting is disabled — it breaks at high particle counts.
    // Sorting particles along a Z-order curve should improve cache locality and cut
    // warp divergence during the tree walk, since neighbouring threads would then
    // traverse similar paths. Suspected cause of the breakage is the subtree
    // insertion in ParallelOctreeGPU::executeTasks; needs further debugging.
    //particles->gpuSort();

    // Rebuild the octree on the GPU (expand → bucket particles into tasks → insert → propagate → prune)
    this->octree->insert(particles);

    // Force pass: one thread per particle walks the tree, opening a node only when
    // it is too close/large to approximate by its center of mass (theta criterion).
    this->forceCalculator->use();
    this->forceCalculator->setInt("numParticles", particles->size());
    this->forceCalculator->setFloat("G", this->G);
    this->forceCalculator->setFloat("theta", this->theta);
    this->forceCalculator->setFloat("squaredSoftening", this->squaredSoftening);
    this->forceCalculator->setInt("fatherTreeNodes", this->octree->getFatherTreeNodes());
    glDispatchCompute((particles->size()+1024-1) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Integration pass: apply the accumulated forces to velocities and positions.
    // The barrier above guarantees all forces are visible before this reads them.
    this->positionCalculator->use();
    this->positionCalculator->setInt("numParticles", particles->size());
    glDispatchCompute(ceil(particles->size() / 64.0), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

}


bool ParticleSolverBHutGPU::usesGPU() {return true;}



ParticleSolverBHutGPU::~ParticleSolverBHutGPU() = default;