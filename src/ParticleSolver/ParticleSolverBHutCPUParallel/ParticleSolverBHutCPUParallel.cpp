// Parallel Barnes-Hut CPU solver.
// Same algorithm as the sequential version, but the octree is built by
// ParallelOctreeCPU (subtrees distributed across threads) and both the force
// traversal and the integration step are spread over cores with OpenMP.

#include "ParticleSolverBHutCPUParallel.h"

#include <iostream>

#include "ParallelOctreeCPU.h"
#include <glm/gtx/norm.hpp>

ParticleSolverBHutCPUParallel::ParticleSolverBHutCPUParallel(float stepSize, float squaredSoft, int n): ParticleSolver() {
    this->squaredSoftening = squaredSoft;
    this->timeStep = stepSize;
    this->G = 1.0f;
    this->octree = new ParallelOctreeCPU(n);
}



void ParticleSolverBHutCPUParallel::updateParticlePositions(ParticleSystem *particles){

    // Recompute the root bounding box from the current particle positions
    octree->reset(particles);

    // Rebuild the tree: particles are bucketed into subtrees that threads fill independently
    octree->insert(particles);

    // Force pass. Dynamic scheduling because the cost of a tree walk varies a lot
    // per particle: those in dense regions descend much deeper than isolated ones.
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < particles->size(); i++) {
        this->computeGravityForce(particles, i);
    }

    // Integration pass. Uniform cost per particle, so a static split is cheapest.
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i<particles->size(); i++){
        particles->updateParticlePosition(i, this->timeStep);
    }

}

// Walks the octree for a single particle, approximating distant clusters by their
// center of mass (the Barnes-Hut criterion is applied inside the octree walk).
void
ParticleSolverBHutCPUParallel::computeGravityForce(ParticleSystem *particles, const unsigned int particleId) {
    particles->getForces()[particleId] = this->octree->computeGravityForce(particles->getPositions()[particleId], this->squaredSoftening, this->G);
}

bool ParticleSolverBHutCPUParallel::usesGPU() {return false;}


float ParticleSolverBHutCPUParallel::getSquaredSoftening() {
    return this->squaredSoftening;
}

ParticleSolverBHutCPUParallel::~ParticleSolverBHutCPUParallel() {
   delete this->octree;
}