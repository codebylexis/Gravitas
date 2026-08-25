// Owns the main loop: advance the simulation, draw, present, handle input.
// Simulation and rendering are deliberately decoupled — the physics step is skipped
// while paused, but drawing continues so the camera stays interactive.

#include "RenderLoop.h"

RenderLoop::RenderLoop(const Window& win, bool showFps, bool vSync) :
    window(win),
    renderTimer(RenderTimer(showFps, vSync)),
    pauseSimulation(true)
{}

RenderLoop::RenderLoop()=default;

void RenderLoop::runLoop(std::shared_ptr<ParticleSimulation> particleSimulation) {

    while (!glfwWindowShouldClose(this->window.getWindow()))
    {
        // Updates delta time and the FPS counter in the window title
        this->renderTimer.updateTime(this->window, this->pauseSimulation);

        // One physics step per frame. The simulation starts paused so the initial
        // particle configuration can be inspected before it evolves.
        if(!this->pauseSimulation){
            particleSimulation->update();
        }

        particleSimulation->draw(renderTimer.getDeltaTime());

        // ======================
        // Swap buffers: Front buffer(render) and back buffer (next render)
        glfwSwapBuffers(this->window.getWindow());
        // ======================

        // ======================
        // Checks if any events are triggered (keys pressed/released, mouse moved etc.) and calls the corresponding functions
        glfwPollEvents();
        // ======================

    }
}

void RenderLoop::setPauseSimulation(bool pause) {
    this->pauseSimulation = pause;
}


int RenderLoop::getIteration() {
    return this->renderTimer.getIteration();
}


RenderLoop::~RenderLoop(){
    this->renderTimer.printFinalStats();
}


