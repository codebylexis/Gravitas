// Brute-force (particle-particle) CPU solver, parallelized with OpenMP.
// Every particle is compared against every other one, so the cost is O(n^2) per
// frame. This is the accuracy reference the Barnes-Hut solvers are checked against.

#include "ParticleSolverCPUParallel.h"
#include <omp.h>
#include <glm/gtx/norm.hpp>
#include <iostream>
ParticleSolverCPUParallel::ParticleSolverCPUParallel(float stepSize, float squaredSoft): ParticleSolver() {
    this->squaredSoftening = squaredSoft;
    this->timeStep = stepSize;
    this->G = 1.0f;
}

void ParticleSolverCPUParallel::updateParticlePositions(ParticleSystem *particles){

    // Every particle costs the same n-1 interactions, so a static split balances evenly.
    // Forces must be fully computed before any position moves, hence the two separate loops.
    #pragma omp parallel for schedule(static) shared(particles)
    for(size_t i =  0; i < particles->size(); i++){
        this->computeGravityForce(particles, i);
    }

    #pragma omp parallel for schedule(static) shared(particles)
    for(size_t i =  0; i<particles->size(); i++){
        particles->updateParticlePosition(i, this->timeStep);
    }
}


// Accumulates the gravitational pull of every other particle on `particleId`.
// Uses Plummer softening: the (r^2 + eps^2)^1.5 denominator keeps the force finite
// when two particles get arbitrarily close, which would otherwise blow up the integrator.
void
ParticleSolverCPUParallel::computeGravityForce(ParticleSystem *particles, const unsigned int particleId) {
    glm::vec4 particlePosition = particles->getPositions()[particleId];
    glm::vec4 totalForce (0.f);

    for(size_t j = 0; j < particles->size(); j++){
        if (particleId != j) {
            const glm::vec4 vector_i_j = particles->getPositions()[j] - particlePosition;
            const float distance_i_j = std::pow(glm::length2(vector_i_j) + this->squaredSoftening, 1.5f);
            totalForce += ((G * particles->getMasses()[j].x) / distance_i_j) * vector_i_j;
        }
    }

    particles->getForces()[particleId] = totalForce;
}

bool ParticleSolverCPUParallel::usesGPU() {return false;}


float ParticleSolverCPUParallel::getSquaredSoftening() {
    return this->squaredSoftening;
}