#include "enums.h"
#include <cstddef>
#include <string>

#ifndef N_BODY_ARGUMENTSPARSER_H
#define N_BODY_ARGUMENTSPARSER_H

// Parses command-line arguments and exposes simulation configuration.
// Prints a usage summary to stdout on every run so the user always sees
// the available options.
//
// Supported flags:
//   -v / -version   : solver version (see Version enum)
//   -n              : number of particles
//   -i / -init      : initialization type (see InitializationType enum)
//   -t / -time-step : simulation time step
//   -s / -softening : squared Plummer softening parameter
//   -f / -file      : path to a particle system file (overrides -i)
class ArgumentsParser {
public:
    ArgumentsParser(int argc, char *argv[]);
    Version getVersion();
    InitializationType getInitializationType();
    size_t getNumParticles();
    float getTimeStep();
    float getSquaredSoftening();
    std::string getFilePath();
private:
    Version version;
    InitializationType init;
    size_t numParticles;
    float timeStep;
    float squaredSoftening;
    std::string filePath;
};


#endif //N_BODY_ARGUMENTSPARSER_H
