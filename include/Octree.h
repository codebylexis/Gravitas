#include "ParticleSystem.h"
#include "Node.h"

#ifndef OCTREE_H
#define OCTREE_H

// Sequential Barnes-Hut octree for 3-D N-body simulation.
//
// Each frame the tree is rebuilt from scratch in three passes:
//   1. reset()     — clear the tree and recompute the root bounding box
//   2. insert()    — insert all particles one by one (average O(n log n))
//   3. propagate() — bottom-up pass to accumulate mass / center-of-mass
//   4. prune()     — remove empty branches so the traversal skips them
//
// Force evaluation calls computeGravityForce() per particle, which walks
// the tree with a stackless linked-list traversal using each node's "next"
// pointer (see Node.h for the packed layout).
class Octree {
    public:
      Octree(){}
      Octree(int n);
      virtual void reset(ParticleSystem* p);
      virtual ~Octree();
      virtual void insert(glm::vec4 pos, glm::vec4 mass, int root);
      virtual void propagate();
      virtual glm::vec4 computeGravityForce(glm::vec4& pos, float squaredSoftening, float G);
      virtual void prune();
      virtual Node *getNodes();
      virtual int getNodeCount();
      virtual int getMaxNodes();
      virtual void setNodes(Node *newNodes);
    protected:
      Node* nodes;       // flat array of all tree nodes
      int *parents;      // indices of internal (parent) nodes, used for bottom-up propagation
      int parentCount;   // number of internal nodes currently in use
      int maxNodes;      // capacity of the nodes array
      int nodeCount;     // number of nodes currently in use
      float theta;       // Barnes-Hut approximation threshold (stored as theta^2)
      virtual void adjustBoundingBox(ParticleSystem* p);
      virtual void insertParticle(glm::vec4 pos, glm::vec4 mass, int i);
      virtual void subdivide(int i);
      virtual int getNextNode(glm::vec4 pos, int i);
};



#endif //OCTREE_H
