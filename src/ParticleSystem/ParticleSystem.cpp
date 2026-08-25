// Storage for the particle state, laid out as a struct-of-arrays.
//
// Positions, velocities, accelerations, masses and forces each live in their own
// contiguous array rather than being interleaved per particle. Solvers usually touch
// one attribute at a time, so this keeps their reads sequential, and each array maps
// directly onto an SSBO the shaders can consume without any repacking.

#include "ParticleSystem.h"
#include "Helpers.h"
#include <algorithm>
using helpers::log;
ParticleSystem::ParticleSystem(const std::vector<Particle> &particles, const bool usesGPU) {
    this->numParticles = particles.size();
    this->velocities = new glm::vec4[this->numParticles]();
    this->accelerations = new glm::vec4[this->numParticles]();
    this->positions = new glm::vec4[this->numParticles]();
    this->masses = new glm::vec4[this->numParticles]();
    this->forces = new glm::vec4[this->numParticles]();


    for (int i = 0; i < this->numParticles; i++) {
        this->velocities[i] = particles[i].velocity;
        this->accelerations[i] = particles[i].acceleration;
        this->positions[i] = particles[i].position;
        this->masses[i] = glm::vec4(particles[i].mass, 0, 0, 0);
        this->forces[i] = glm::vec4(0.f);
    }

    this->createBuffers(usesGPU);
	this->compileShaders();
}

void ParticleSystem::compileShaders() {
    mortonShader = std::make_unique<ComputeShader>(std::string("../src/shaders/ComputeShaders/morton.glsl"));
    rearrangeParticlesShader = std::make_unique<ComputeShader>(std::string("../src/shaders/ComputeShaders/rearrange.glsl"));
	bitonicSortShader = std::make_unique<ComputeShader>(std::string("../src/shaders/ComputeShaders/bitonicSort.glsl"));
}

ParticleSystem::ParticleSystem(ParticleSystem * other) {
    this->numParticles = other->size();
    this->velocities = new glm::vec4[this->numParticles]();
    this->accelerations = new glm::vec4[this->numParticles]();
    this->positions = new glm::vec4[this->numParticles]();
    this->masses = new glm::vec4[this->numParticles]();
    this->forces = new glm::vec4[this->numParticles]();

    for (int i = 0; i < this->numParticles; i++) {
        this->velocities[i] = other->getVelocities()[i];
        this->accelerations[i] = other->getAccelerations()[i];
        this->positions[i] = other->positions[i];
        this->masses[i] = other->masses[i];
        this->forces[i] = other->forces[i];
    }
}


/**
 * Picks a buffer strategy based on where the solver runs.
 * GPU solvers keep their data device-side; CPU solvers need host-visible memory.
 */
void ParticleSystem::createBuffers(bool usesGPU) {
    if (usesGPU) {
        this->configureGpuBuffers();
    }
    else {
        this->configureCpuBuffers();
    }
}

/**
 * Rounds up to the next power of two by smearing the highest set bit down across
 * all lower bits, then incrementing. Needed because bitonic sort only works on
 * power-of-two sized inputs.
 */
unsigned int ParticleSystem::nextPowerOfTwo(unsigned int n){
    if (n == 0) return 1; // Or handle as needed
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;

}

/**
 * Uploads particle state to device-local buffers. Nothing is host-mapped: GPU solvers
 * read and write these in place, and the renderer draws from the same positions SSBO.
 */
void ParticleSystem::configureGpuBuffers() {
	this->paddedNumParticles = nextPowerOfTwo(this->numParticles);
    positions_SSBO.createBufferData(sizeof(glm::vec4) * this->size(), this->getPositions(), 0, GL_DYNAMIC_COPY);
    velocities_SSBO.createBufferData(sizeof(glm::vec4) * this->size(), this->getVelocities(), 1, GL_DYNAMIC_COPY);
    accelerations_SSBO.createBufferData(sizeof(glm::vec4) * this->size(), this->getAccelerations(), 2, GL_DYNAMIC_COPY);
    masses_SSBO.createBufferData(sizeof(glm::vec4) * this->size(), this->getMasses(), 3, GL_STATIC_DRAW);
    forces_SSBO.createBufferData(sizeof(glm::vec4) * this->size(), this->getForces(), 4, GL_DYNAMIC_COPY);
    mortonBuffer.createBufferData(sizeof(glm::uvec2) * paddedNumParticles, nullptr, 16, GL_DYNAMIC_COPY);
    positionsCopy.createBufferData(sizeof(glm::vec4) * this->size(), nullptr, 17, GL_DYNAMIC_COPY);
    velocitiesCopy.createBufferData(sizeof(glm::vec4) * this->size(), nullptr, 18, GL_DYNAMIC_COPY);
}

/**
 * Creates persistent mapped shader storage buffer objects.
 * The mapping stays valid for the lifetime of the buffer, so a CPU solver can write
 * straight into memory the renderer reads — no per-frame upload or re-mapping.
 * COHERENT makes those writes visible to the GPU without an explicit flush.
 */
void ParticleSystem::configureCpuBuffers() {
    GLbitfield bufferStorageFlags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;

    positions_SSBO.createBufferStorage(sizeof(glm::vec4) * this->size(), this->getPositions(), 0, bufferStorageFlags);
    glm::vec4* positions = (glm::vec4*)positions_SSBO.mapBufferRange(0, sizeof(glm::vec4) * this->size(), bufferStorageFlags);
    this->setPositions(positions);

    velocities_SSBO.createBufferStorage(sizeof(glm::vec4) * this->size(), this->getVelocities(), 1, bufferStorageFlags);
    glm::vec4* velocities = (glm::vec4*)velocities_SSBO.mapBufferRange(0, sizeof(glm::vec4) * this->size(), bufferStorageFlags);
    this->setVelocities(velocities);

    accelerations_SSBO.createBufferStorage(sizeof(glm::vec4) * this->size(), this->getAccelerations(), 2, bufferStorageFlags);
    glm::vec4* accelerations = (glm::vec4*)accelerations_SSBO.mapBufferRange(0, sizeof(glm::vec4) * this->size(), bufferStorageFlags);
    this->setAccelerations(accelerations);
}

// Sort the particles in the GPU using the z-order curve
// Why? Improve cache locality, memory access patterns and reduce warp divergence
void ParticleSystem::gpuSort() {
	// Rearranging is a scatter, so it can't be done in place. Alternate between the
	// real buffers and the copies each call instead of copying back every frame.
	static bool pingPong = true;

	OpenGLBuffer* readPositions = pingPong ? &positions_SSBO : &positionsCopy;
	OpenGLBuffer* readVelocities = pingPong ? &velocities_SSBO : &velocitiesCopy;
	OpenGLBuffer* writePositions = pingPong ? &positionsCopy : &positions_SSBO;
	OpenGLBuffer* writeVelocities = pingPong ? &velocitiesCopy : &velocities_SSBO;

    // Calculate morton codes
    mortonShader->use();
	readPositions->bindBufferBase(0);
	readVelocities->bindBufferBase(1);
	mortonBuffer.bindBufferBase(16);
    mortonShader->setInt("actualNumParticles", this->numParticles);
	mortonShader->setInt("paddedNumParticles", this->paddedNumParticles);
	glDispatchCompute((this->paddedNumParticles + 256 - 1) / 256, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  

	// Sort morton codes using bitonic sort (implement radix sort for better performance) 
	bitonicSortShader->use();
    readPositions->bindBufferBase(0);
    readVelocities->bindBufferBase(1);
    mortonBuffer.bindBufferBase(16);
	bitonicSortShader->setInt("numParticles", this->paddedNumParticles);
    
    // Bitonic sort requires numParticles to be a power of 2 for the simplest
    // implementation. If not, you either need to pad your buffers/particle count
    // to the next power of two OR modify the shader/dispatch logic to handle
    // the exact size (more complex bounds checking).

    const int N = this->paddedNumParticles; // Or use next power of 2 if padding

    for (unsigned int k = 2; k <= N; k <<= 1) { // k = size of sequences being merged
        bitonicSortShader->setInt("k", k);
        for (unsigned int j = k >> 1; j > 0; j >>= 1) { // j = comparison distance
            if (j < 1) continue; // Should not happen with k >= 2
            bitonicSortShader->setInt("j", j);

            glDispatchCompute((this->paddedNumParticles + 256 - 1) / 256, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }
   
	
	// Rearrange particles based on sorted morton codes
	rearrangeParticlesShader->use();
    readPositions->bindBufferBase(0);
    readVelocities->bindBufferBase(1);
    mortonBuffer.bindBufferBase(16);
	writePositions->bindBufferBase(17);
	writeVelocities->bindBufferBase(18);
	rearrangeParticlesShader->setInt("numParticles", this->numParticles);
	glDispatchCompute((this->numParticles + 256 - 1) / 256, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	// Swap roles: the buffers just written become the canonical ones at bindings 0/1,
	// so the rest of the frame reads sorted data without knowing a sort happened.
	pingPong = !pingPong;
	writePositions->bindBufferBase(0);
	writeVelocities->bindBufferBase(1);
	readPositions->bindBufferBase(17);
	readVelocities->bindBufferBase(18);
}


// Setters take ownership of the incoming array and free the one they replace.
// Used when swapping heap arrays for persistently mapped buffer memory.
void ParticleSystem::setAccelerations(glm::vec4 *newAccelerations) {
    delete [] this->accelerations;
    this->accelerations = newAccelerations;
}

void ParticleSystem::setMasses(glm::vec4 *newMasses) {
    delete [] this->masses;
    this->masses = newMasses;
}

void ParticleSystem::setForces(glm::vec4 *newForces) {
    delete [] this->forces;
    this->forces = newForces;
}

void ParticleSystem::setPositions(glm::vec4 *newPositions) {
    delete [] this->positions;
    this->positions = newPositions;
}

void ParticleSystem::setVelocities(glm::vec4 *newVelocities) {
    delete [] this->velocities;
    this->velocities = newVelocities;
}

std::ostream& operator<<(std::ostream& os, const ParticleSystem& system) {
    os << "Particle System with " << system.numParticles << " particles:" << std::endl;
    for (unsigned int i = 0; i < system.numParticles; ++i) {
        os << "Particle ID: " << i << std::endl;
        os << "Position: (" << system.positions[i].x << ", " << system.positions[i].y << ", " << system.positions[i].z << ")" << std::endl;
        os << "Velocity: (" << system.velocities[i].x << ", " << system.velocities[i].y << ", " << system.velocities[i].z << ")" << std::endl;
        os << "Acceleration: (" << system.accelerations[i].x << ", " << system.accelerations[i].y << ", " << system.accelerations[i].z << ")" << std::endl;
        os << "Mass: " << system.masses[i].x << std::endl;
    }
    return os;
}