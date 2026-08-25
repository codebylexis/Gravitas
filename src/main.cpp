// Gravitas — N-body gravitational simulation
// Supports multiple solver backends (CPU sequential/parallel, GPU compute shaders, Barnes-Hut variants).
// The solver and particle initializer are selected via command-line arguments and injected into
// ParticleSimulation, which owns the main update/draw loop.

#include <memory>
#include "ArgumentsParser.h"

#include "ParticleSystemCubeInitializer.h"
#include "ParticleSystemGalaxyInitializer.h"
#include "ParticleSystemLagrange.h"
#include "ParticleSystemSphere.h"
#include "ParticleSystemBall.h"
#include "ParticleSystemCubeSurface.h"
#include "ParticleSystemFile.h"


#include "ParticleSolverCPUSequential.h"
#include "ParticleSolverCPUParallel.h"
#include "ParticleSolverGPU.h"
#include "ParticleSolverBHutCPUSeq.h"
#include "ParticleSolverBHutCPUParallel.h"
#include "ParticleSolverBHutGPU.h"



#include "WindowInputManager.h"

int main(int argc, char *argv[])
{
    // Parse command-line flags (-v, -n, -i, -t, -s, -f)
    ArgumentsParser args(argc, argv);

    // The simulation world is a 5×5×5 cube centered at the origin
    glm::vec3 worldDimensions(5.f);

    // Using pixels
    glm::vec2 windowDim(1300, 750);
    Window window(windowDim, "Gravitas");

    // Second arg: print FPS stats; third arg: enable VSync
    RenderLoop renderLoop(window, true, true);

    // --- Particle initializer selection ---
    // Each initializer generates the starting positions, velocities, and masses
    // for a specific arrangement (galaxy disk, cube, sphere, etc.)
    std::unique_ptr<ParticleSystemInitializer> particleSystemInitializer;

    std::string filePath = args.getFilePath();

    switch (args.getInitializationType()) {
        case InitializationType::GALAXY:
            particleSystemInitializer = std::make_unique<ParticleSystemGalaxyInitializer>(args.getNumParticles());
            break;
        case InitializationType::CUBE:
            particleSystemInitializer = std::make_unique<ParticleSystemCubeInitializer>(args.getNumParticles());
            break;
        case InitializationType::LAGRANGE:
            // Fixed 3-body Lagrange point configuration (equilateral triangle, zero velocity)
            particleSystemInitializer = std::make_unique<ParticleSystemLagrange>();
            break;
        case InitializationType::SPHERE:
            particleSystemInitializer = std::make_unique<ParticleSystemSphere>(args.getNumParticles());
            break;
        case InitializationType::BALL:
            particleSystemInitializer = std::make_unique<ParticleSystemBall>(args.getNumParticles());
            break;
        case InitializationType::CUBE_SURFACE:
            particleSystemInitializer = std::make_unique<ParticleSystemCubeSurface>(args.getNumParticles());
            break;
        case InitializationType::SYSTEM_FILE:
            particleSystemInitializer = std::make_unique<ParticleSystemFile>(filePath);
            break;
        default:
            exit(EXIT_FAILURE);
    }

    std::shared_ptr<ParticleSimulation> particleSimulation;

    // Shader paths are relative to the build directory (cmake out-of-source build assumed)
    std::string positionsCalculatorPath;
    std::string forceCalculatorPath;

    // --- Solver selection ---
    // CPU solvers run on the host; GPU solvers dispatch OpenGL compute shaders.
    // Barnes-Hut solvers build an octree each frame and approximate distant clusters,
    // reducing per-frame work from O(n^2) to O(n log n).
    switch (args.getVersion()){
        case Version::PP_CPU_SEQUENTIAL:
            // Brute-force O(n^2) on a single CPU thread — reference implementation
            particleSimulation = std::make_shared<ParticleSimulation>(
                std::move(particleSystemInitializer),
                std::move(
                    std::make_unique<ParticleSolverCPUSequential>(args.getTimeStep(), args.getSquaredSoftening())
                ),
                worldDimensions,
                windowDim
            );
            break;
        case Version::PP_CPU_PARALLEL:
            // Brute-force O(n^2) parallelized across CPU cores via OpenMP
            particleSimulation = std::make_shared<ParticleSimulation>(
                std::move(particleSystemInitializer),
                std::move(
                    std::make_unique<ParticleSolverCPUParallel>(args.getTimeStep(), args.getSquaredSoftening())
                ),
                worldDimensions,
                windowDim
            );
            break;
        case Version::PP_GPU_PARALLEL:
            // Brute-force O(n^2) dispatched as a GPU compute shader (one thread per particle)
            positionsCalculatorPath = "../src/shaders/ComputeShaders/updateParticles.glsl";
            forceCalculatorPath = "../src/shaders/ComputeShaders/forceCalculation.glsl";
            particleSimulation = std::make_shared<ParticleSimulation>(
                std::move(particleSystemInitializer),
                std::move(
                    std::make_unique<ParticleSolverGPU>(args.getTimeStep(), args.getSquaredSoftening(), positionsCalculatorPath, forceCalculatorPath)
                ),
                worldDimensions,
                windowDim
            );
            break;
        case Version::PP_GPU_OPTIMIZED:
            positionsCalculatorPath = "../src/shaders/ComputeShaders/updateParticles.glsl";
            forceCalculatorPath = "../src/shaders/ComputeShaders/forceCalculationOptimized.glsl";
            {
                // Tile size for the shared-memory tiled N-body kernel (NVIDIA GPU Gems 3, Ch. 31).
                // Each thread block loads kOptimizedBlockSize positions into shared memory,
                // reducing global memory reads from O(n^2) to O(n^2 / kOptimizedBlockSize).
                constexpr double kOptimizedBlockSize = 320.0;
                particleSimulation = std::make_shared<ParticleSimulation>(
                    std::move(particleSystemInitializer),
                    std::move(
                        std::make_unique<ParticleSolverGPU>(kOptimizedBlockSize, args.getTimeStep(), args.getSquaredSoftening(), positionsCalculatorPath, forceCalculatorPath)
                    ),
                    worldDimensions,
                    windowDim
                );
            }
            break;

        case Version::BARNES_HUT_CPU_SEQ:
            // O(n log n) Barnes-Hut, sequential: builds an octree then walks it per particle
            particleSimulation = std::make_shared<ParticleSimulation>(
                std::move(particleSystemInitializer),
                std::move(
                    std::make_unique<ParticleSolverBHutCPUSeq>(args.getTimeStep(), args.getSquaredSoftening(), args.getNumParticles())
                ),
                worldDimensions,
                windowDim
            );
            break;

        case Version::BARNES_HUT_CPU_PARALLEL:
            // O(n log n) Barnes-Hut with OpenMP: tree build and force traversal both parallelized
            particleSimulation = std::make_shared<ParticleSimulation>(
                std::move(particleSystemInitializer),
                std::move(
                    std::make_unique<ParticleSolverBHutCPUParallel>(args.getTimeStep(), args.getSquaredSoftening(), args.getNumParticles())
                    ),
                worldDimensions,
                windowDim
            );
            break;

        case Version::BARNES_HUT_GPU_PARALLEL:
            // O(n log n) Barnes-Hut fully on GPU: tree built via compute shaders, force traversal
            // uses shared memory to cache upper tree levels (fatherTreeNodes) for fast access
            positionsCalculatorPath = "../src/shaders/ComputeShaders/updateParticles.glsl";
            forceCalculatorPath = "../src/shaders/ComputeShaders/forceCalculateBarnesHut.glsl";
            particleSimulation = std::make_shared<ParticleSimulation>(
                std::move(particleSystemInitializer),
                std::move(
                    std::make_unique<ParticleSolverBHutGPU>(args.getTimeStep(), args.getSquaredSoftening(), args.getNumParticles(), positionsCalculatorPath, forceCalculatorPath)
                ),
                worldDimensions,
                windowDim
            );
            break;
    }

    // Keyboard/mouse input drives camera and pause/resume
    WindowInputManager windowInputManager(&window, &renderLoop, particleSimulation);

    renderLoop.runLoop(particleSimulation);

}
