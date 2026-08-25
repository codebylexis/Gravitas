// Multithreaded octree build for the CPU Barnes-Hut solver.
//
// The tree is split into two tiers so threads rarely contend:
//   * A shallow "father" tree of maxNodes nodes, subdivided eagerly down to
//     maxDepth. Its deepest level defines totalTasks buckets.
//   * One independent subtree per bucket, each with its own node arena, node
//     counter and parent list. A thread owns a whole subtree, so insertion into
//     it needs no locking at all.
// Locks are only taken while bucketing particles into tasks; everything after
// that is embarrassingly parallel until the father tree is propagated.

#include "ParallelOctreeCPU.h"
#include <omp.h>
#include <iostream>
#include <glm/gtx/norm.hpp>


ParallelOctreeCPU::ParallelOctreeCPU(const int n): Octree(n) {
    this->numThreads = omp_get_max_threads();
    this->maxDepth = -1;

    // Octree allocates a flat node arena for the sequential layout; this class needs
    // a different one (father tree + per-subtree arenas), so drop it and re-allocate below.
    delete[] this->nodes;

    // Grow the father tree one level at a time while a full level still fits in n.
    // Capped at depth 3 (8^3 = 512 tasks), which is enough parallelism for any
    // reasonable core count without wasting nodes on a mostly-empty upper tree.
    this->maxNodes = 0;
    int i = 0;
    while (maxNodes < n && maxDepth < 3) {
        int tmp = maxNodes + std::pow(8, i);
        if (tmp >= n) break;
        maxNodes = tmp;
        maxDepth++;
        i++;
    }
    std::cout << maxNodes << " " << maxDepth << std::endl;

    // Worst-case arena size for one subtree. Generous on purpose: a single bucket can
    // receive far more than its even share when the particle distribution is clumped.
    this->maxNodesPerSubtree = n/2.0;

    // One lock per father-tree node, used only while bucketing particles into tasks
    this->nodeLocks.resize(maxNodes);

    #pragma omp parallel for
    for (int i = 0; i < maxNodes; ++i) {
        omp_init_lock(&nodeLocks[i]);
    }

    // Each leaf of the father tree becomes one independently-buildable task
    this->totalTasks = std::pow(8, maxDepth);

    // Single allocation holding the father tree followed by all subtree arenas,
    // so a node index is directly comparable across both tiers
    this->nodes = new Node[this->maxNodesPerSubtree * this->totalTasks + this->maxNodes];


    std::cout << getMaxNodes() << std::endl;

    this->tasks = new Task[totalTasks];

    // Per-subtree bookkeeping, indexed by task id so threads never share a cache line's worth of state
    nodeCounts = new int[this->totalTasks];
    subTreeParents = new int[this->totalTasks*this->maxNodesPerSubtree];
    parentCounts = new int[this->totalTasks];

    this->maxParticlesPerTask = maxNodesPerSubtree * 0.6;
    std::cout << maxNodesPerSubtree << " " << totalTasks << " " << maxParticlesPerTask<< std::endl;

    // Flat 2D array: task i owns the slice [i * maxParticlesPerTask, (i+1) * maxParticlesPerTask)
    taskParticles = new int[this->totalTasks * maxParticlesPerTask];

    this->resetArrays();
}

ParallelOctreeCPU::~ParallelOctreeCPU() {
    delete[] tasks;
    delete[] nodeCounts;
    delete[] subTreeParents;
    delete[] parentCounts;
    delete[] taskParticles;

    for (int i = 0; i < maxNodes; ++i) {
        omp_destroy_lock(&nodeLocks[i]);
    }
}

/**
 * Establishes the bounding box of the root node
 * O(n)
 * @param p Particles
 */
void ParallelOctreeCPU::reset(ParticleSystem *p) {
    float min_x = std::numeric_limits<float>::max();
    float min_y = min_x;
    float min_z = min_x;
    float max_x = std::numeric_limits<float>::min();
    float max_y = max_x;
    float max_z = max_x;

    // Find the max and min positions
    #pragma omp parallel for reduction(min: min_x, min_y, min_z) reduction(max: max_x, max_y, max_z)
    for (int i = 0;  i < p->size(); ++i){
        const glm::vec4 pos = p->getPositions()[i];

        if (pos.x < min_x){
            min_x = pos.x;
        }
        if (pos.y < min_y){
            min_y = pos.y;
        }
        if (pos.z < min_z){
            min_z = pos.z;
        }
        if (pos.x > max_x){
            max_x = pos.x;
        }
        if (pos.y > max_y){
            max_y = pos.y;
        }
        if (pos.z > max_z){
            max_z = pos.z;
        }
    }

    // Set the bounding box of the root node
    this->nodes[0].setBoundingBox(glm::vec4(min_x, min_y, min_z, 0.f), glm::vec4(max_x, max_y, max_z, 0.f));

    Node &root = this->nodes[0];
    root.setFirstChild(-1);
    root.setMass(glm::vec4(-1.f, 0.f, 0.f, 0.f));
    this->nodeCount = 1;
    this->parentCount = 0;
    root.setNext(-1);
}

/**
 * Builds the whole tree for the current frame, in four phases:
 *   1. Eagerly subdivide the father tree down to maxDepth.
 *   2. Bucket every particle into the task owning its deepest father-tree cell.
 *   3. Build each subtree in parallel, one thread per task, and propagate it.
 *   4. Stitch the subtree roots back onto the father tree, then propagate and prune globally.
 * @param p particles
 */
void ParallelOctreeCPU::insert(ParticleSystem *p) {

    int root = 0;

    // Phase 1: subdivide unconditionally. Unlike the sequential octree, the upper
    // levels are always fully expanded so the task layout is identical every frame.
    while (nodeCount < maxNodes) {
        Octree::subdivide(root);
        root+=1;
    }


    // Phase 2: assign each particle to a task by descending to maxDepth.
    // The lock is per father-tree node, so threads only serialize when two particles
    // land in the same cell — and only for the length of an array append.
    #pragma omp parallel for
    for (int j = 0; j < p->size(); j++) {
        const glm::vec4 pos = p->getPositions()[j];
        int i = 0;
        int depth = 0;
        while(depth < maxDepth) {
            // Go down until you are at max depth
            // Traverse the tree
            i = this->getNextNode(pos, i);
            depth ++;
        }
        omp_set_lock(&nodeLocks[i]);  // Lock only for this node
        taskParticles[i%totalTasks * maxParticlesPerTask + this->tasks[i%totalTasks].totalParticles] = j;
        this->tasks[i%totalTasks].totalParticles++;
        this->tasks[i%totalTasks].root = i;
        omp_unset_lock(&nodeLocks[i]);  // Unlock after insertion
    }


    // Phase 3: build the subtrees. Dynamic scheduling because bucket occupancy is
    // very uneven — a dense cluster can leave one task with most of the particles.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < totalTasks; ++i) {
        // Execute task
        Task &task = this->tasks[i];
        if (task.root < 0) continue;

        int root = getSubtree(i);
        Node &oct = this->nodes[root];


        // Reset the subtree root: it inherits the bounding box of the father-tree
        // leaf it hangs off, and starts as an empty leaf
        oct.setBoundingBox(this->nodes[task.root].getMinBoundary(), this->nodes[task.root].getMaxBoundary());
        oct.setFirstChild(-1);
        oct.setMass(glm::vec4(-1.f, 0.f, 0.f, 0.f));
        nodeCounts[i] = 1;
        parentCounts[i] = 0;
        oct.setNext(-1);


        if (task.totalParticles <= 0) continue;

        // Graft the subtree onto the father tree, splicing it into the traversal chain
        oct.setNext(this->nodes[task.root].getNext());
        this->nodes[task.root].setFirstChild(root);

        for (int j = 0; j < task.totalParticles; j++) {
            const int particleId = taskParticles[i * maxParticlesPerTask + j];
            this->insert(p->getPositions()[particleId], p->getMasses()[particleId], i);
        }

        // Roll masses up within this subtree only — no other thread touches these nodes
        this->propagate(i);

    }


    // Phase 4: register the non-empty father-tree leaves as parents so the global
    // propagate pass picks up the masses their subtrees just computed
    for(int i = maxNodes-totalTasks; i < maxNodes; i++){
        if(this->tasks[i%totalTasks].totalParticles > 0){
            this->parents[parentCount] = i;
            parentCount++;
        }
    }

    this->propagate();

    this->prune();


}


/**
 * Rewires the father tree so the force traversal skips empty nodes entirely.
 * Each occupied parent points at its first occupied child, and occupied siblings are
 * chained through their `next` links — turning the tree walk into a pointerless
 * linked-list descent with no emptiness checks in the inner loop.
 */
void ParallelOctreeCPU::prune() {
    // Sequential over the father tree: a parent's rewiring depends on links that the
    // level below may still be adjusting, so this tier stays single-threaded.
    for (int i = 0; i < maxNodes-totalTasks; i++) {
        const int parentIndex = parents[i];
        Node &parent = this->nodes[parentIndex];
        if(!parent.isOccupied()){
            parent.setFirstChild(parent.getNext());
            continue;
        }

        int firstChild = -1;
        int lastChild = -1;


        for (int j = 0; j < 8; j++) {
            const int childIndex = parent.getFirstChild()+ j;
            if (this->nodes[childIndex].isOccupied()) {
                if (firstChild == -1) {
                    firstChild = childIndex;
                    lastChild = childIndex;
                    continue;
                }
                this->nodes[lastChild].setNext(childIndex);
                lastChild = childIndex;
            }

        }
        this->nodes[lastChild].setNext(parent.getNext());
        parent.setFirstChild(firstChild);
    }

    // Link each father-tree leaf's subtree root to whatever follows the leaf,
    // so a walk that descends into a subtree can climb back out to the right sibling
    #pragma omp parallel for
    for(int i = maxNodes-totalTasks; i < maxNodes; i++){
        if(this->nodes[i].isOccupied()){
            this->nodes[this->nodes[i].getFirstChild()].setNext(this->nodes[i].getNext());
        }
    }


    // Subtrees are disjoint, so they prune in parallel with no synchronization
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < totalTasks; i++) {
        Task& task = this->tasks[i];
        if (task.totalParticles <= 0 || task.root < 0) continue;

        this->prune(i);
        // Reset the task so the next frame starts from a clean bucket
        task.reset();
    }

}

/**
 * @return total capacity of the node arena: father tree plus every subtree
 */
int ParallelOctreeCPU::getMaxNodes() {
    return this->maxNodes + this->maxNodesPerSubtree * this->totalTasks;
}

/**
 * Rolls mass and center of mass up the father tree, deepest level first, so every
 * internal node ends up summarizing the cluster beneath it. This is what lets the
 * force walk approximate a whole distant subtree with a single interaction.
 */
void ParallelOctreeCPU::propagate() {
    // Bottom level: each father-tree leaf just copies the totals its subtree computed
    #pragma omp parallel for
    for(int i = parentCount -1; i >= maxNodes-totalTasks; i--){
        const int parentIndex = parents[i];
        const int childIndex = this->nodes[parentIndex].getFirstChild();
        if (!this->nodes[childIndex].isOccupied())  continue; // The leaf is empty
        this->nodes[parentIndex].setCenterOfMass(this->nodes[childIndex].getCenterOfMass());
        this->nodes[parentIndex].setMass(glm::vec4(this->nodes[childIndex].getMass(), 0.f, 0.f, 0.f));
    }

    // Upper levels: walked backwards so children are always finalized before their
    // parent reads them. Sequential for the same reason.
    for (int i = maxNodes-totalTasks - 1; i >= 0; i--) {
        // Mass-weighted average of the children gives the parent's center of mass
        const int parentIndex = parents[i];
        glm::vec4 centerOfMass(0.f);
        glm::vec4 mass(0.f);

        for (int j = 0; j < 8; ++j) {
            const int childIndex = this->nodes[parentIndex].getFirstChild() + j;
            if (!this->nodes[childIndex].isOccupied())  continue; // The leaf is empty
            centerOfMass += this->nodes[childIndex].getCenterOfMass() * this->nodes[childIndex].getMass();
            mass.x += this->nodes[childIndex].getMass();

        }
        this->nodes[parentIndex].setCenterOfMass(centerOfMass/mass.x);
        this->nodes[parentIndex].setMass(mass);
    }
}

/**
 * Same empty-node rewiring as prune(), applied to one subtree.
 * Called from a single thread that owns the subtree, so it needs no locking.
 * @param subTreeId task id of the subtree to prune
 */
void ParallelOctreeCPU::prune(int subTreeId) {
    const int totalParentNodes = parentCounts[subTreeId];

    for (int i = 0; i < totalParentNodes; i++) {
        const int parentIndex = subTreeParents[i+subTreeId*this->maxNodesPerSubtree];
        Node &parent = this->nodes[parentIndex];
        if(!parent.isOccupied()){
            parent.setFirstChild(parent.getNext());
            continue;
        }
        int firstChild = -1;
        int lastChild = -1;


        for (int j = 0; j < 8; j++) {
            const int childIndex = parent.getFirstChild()+ j;
            if (this->nodes[childIndex].isOccupied()) {
                if (firstChild == -1) {
                    firstChild = childIndex;
                    lastChild = childIndex;
                    continue;
                }
                this->nodes[lastChild].setNext(childIndex);
                lastChild = childIndex;
            }

        }
        this->nodes[lastChild].setNext(parent.getNext());
        parent.setFirstChild(firstChild);
    }
}

/**
 * Rolls masses up one subtree. Parents were appended to subTreeParents in the order
 * they were created, i.e. top-down, so iterating backwards visits children first.
 * @param subTreeId task id of the subtree to propagate
 */
void ParallelOctreeCPU::propagate(int subTreeId) {
    const int totalParentNodes = parentCounts[subTreeId];


    for (int i = totalParentNodes - 1; i >= 0; i--) {

        // Compute the center of mass and mass of the current parent node
        int parentIndex = subTreeParents[i+subTreeId*this->maxNodesPerSubtree];
        glm::vec4 centerOfMass(0.f);
        glm::vec4 mass(0.f);

        Node &parent = this->nodes[parentIndex];


        for (int j = 0; j < 8; ++j) {
            const int childIndex = parent.getFirstChild() + j;
            Node &child = this->nodes[childIndex];
            if (!child.isOccupied())  continue; // The leaf is empty
            centerOfMass += child.getCenterOfMass() * child.getMass();
            mass.x += child.getMass();
        }
        parent.setCenterOfMass(centerOfMass/mass.x);
        parent.setMass(mass);
    }
}


/**
 * Resets the tasks
 */
void ParallelOctreeCPU::resetArrays() {
#pragma omp parallel for 
    for (int i = 0; i < totalTasks; ++i) {
        tasks[i].reset();    
    }
}

/**
 * Inserts one particle into a subtree, subdividing as needed.
 * Standard octree insertion: an empty leaf takes the particle directly, an occupied
 * leaf splits and both particles move down, an internal node is descended into.
 * @param pos particle position
 * @param mass particle mass
 * @param subTreeId task id of the subtree that owns this particle
 */
void ParallelOctreeCPU::insert(glm::vec4 pos, glm::vec4 mass, int subTreeId) {
    // Start at the root of the octree
    int i = getSubtree(subTreeId);

    // Keep the root node
    const int root = i; 

    // Traverse the tree
    while(i < i+maxNodesPerSubtree){
        Node &node = this->nodes[i];
   
        // Case 1: The node is an empty leaf
        if(node.isLeaf() && !node.isOccupied()){
            // Proceed to insert the particle
            this->insertParticle(pos, mass, i);
            return;
        }

        // Case 2: The node is an occupied leaf.
        if(node.isLeaf() && node.isOccupied()){
            // Get the current particle's position and mass
            glm::vec4 pos2 = node.getCenterOfMass();
            glm::vec4 mass2 = glm::vec4(node.getMass(), 0.f, 0.f, 0.f);

            // Subdivide the current node
            this->subdivide(i, nodeCounts[subTreeId]+root, subTreeId);

            // Get child indexes
            int childIndex1 = this->getNextNode(pos, i);
            int childIndex2 = this->getNextNode(pos2, i);
            const float dist = glm::length2(pos2-pos);

            int count = 0;
            // If both particles go to the same child node we need to keep subdividing
            while (childIndex1 == childIndex2) {

                // Edge case: two particles at (nearly) the same position would subdivide
                // forever, so merge them into one node. The depth cap is a second guard
                // against float precision stalling the descent.
                if (dist < 1e-3 || count++ == 10) {
                    this->insertParticle(pos, glm::vec4(mass2.x + mass.x, 0.f, 0.f, 0.f), childIndex1);  // In this case, insert both particles anyway
                    return;
                }

                // Keep subdividing
                this->subdivide(childIndex1, nodeCounts[subTreeId]+root, subTreeId);


                // Recalculate child indexes after further subdivisions
                childIndex1 = this->getNextNode(pos, childIndex1);
                childIndex2 = this->getNextNode(pos2, childIndex2);
            }

            // Once particles are in different children, insert them
            this->insertParticle(pos, mass, childIndex1);
            this->insertParticle(pos2, mass2, childIndex2);

            return;
        }

        // Case 3: The node is internal
        // Traverse the tree
        i = this->getNextNode(pos, i);
    }
}


/**
 * Splits a node into 8 octants allocated contiguously from the subtree's arena.
 * The child index is a 3-bit code (z,y,x), so bit j of the loop counter directly
 * selects which half of each axis that child covers.
 * @param i node to subdivide
 * @param firstChild arena index where the 8 children are placed
 * @param subTreeId task id of the owning subtree
 */
void ParallelOctreeCPU::subdivide(int i, int firstChild, int subTreeId) {
    // Set the first child to the current node count
    this->nodes[i].setFirstChild(firstChild);

    // Mark this node as a parent
    this->subTreeParents[subTreeId * this->maxNodesPerSubtree + this->parentCounts[subTreeId]] = i;
    this->parentCounts[subTreeId]++;

    // Create the next child nodes
    this->nodeCounts[subTreeId] += 8;

    const glm::vec4 maxB = this->nodes[i].getMaxBoundary();
    const glm::vec4 minB = this->nodes[i].getMinBoundary();

    const float width = maxB.x - minB.x;
    const float height = maxB.y - minB.y;
    const float depth = maxB.z - minB.z;

    int currentChild = firstChild;

    for (int j = 0; j < 8; ++j) {
        // Create the boundaries for the new node
        glm::vec4 minBoundary(0.f);
        glm::vec4 maxBoundary(0.f);

        // Check the 3rd bit
        if (j & 4) {
            minBoundary.z = minB.z;
            maxBoundary.z = minB.z + (depth/2.f);
        }
        else {
            minBoundary.z = minB.z + (depth/2.f);
            maxBoundary.z = maxB.z;
        }

        // Check the second bit
        if (j & 2) {
            minBoundary.y = minB.y;
            maxBoundary.y = minB.y + (height/2.f);
        }
        else {
            minBoundary.y = minB.y + (height/2.f);
            maxBoundary.y = maxB.y;
        }

        // Check the first bit
        if (j & 1) {
            minBoundary.x = minB.x;
            maxBoundary.x = minB.x + (width/2.f);
        }
        else {
            minBoundary.x = minB.x + (width/2.f);
            maxBoundary.x = maxB.x;
        }


        this->nodes[currentChild].createEmptyNode(minBoundary, maxBoundary);

        // Set the next node
        if (j < 7) {
            // The last child is different
            this->nodes[currentChild].setNext(firstChild+ j + 1);
        }

        // Next child node
        currentChild = firstChild + j + 1;
    }

    // The last child node points to the parent's next
    this->nodes[firstChild+7].setNext(this->nodes[i].getNext());
}

/**
 * @param treeId
 * @return root of the subtree
 */
int ParallelOctreeCPU::getSubtree(int treeId) {
    return this->maxNodes + this->maxNodesPerSubtree * treeId;
}