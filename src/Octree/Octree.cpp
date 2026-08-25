// Sequential Barnes-Hut octree implementation.
// See Octree.h for a description of the overall algorithm and data layout.

#include "Octree.h"
#include <iostream>
#include <glm/gtx/norm.hpp>

/**
 * Constructor
 * @param n number of particles
 */
Octree::Octree(int n){
  // For small n the tree can be very deep, so we over-allocate more aggressively
  this->maxNodes = n < 200 ? n*200 : n*8;
  this->nodes = new Node[this->maxNodes];
  this->parents = new int[this->maxNodes];
  this->nodeCount = 0;
  this->parentCount = 0;
  this->theta = 0.25f; // theta^2, where theta = 0.5 is the Barnes-Hut opening angle criterion
}

/**
 * Establishes the bounding box of the root node
 * O(n)
 * @param p Particles
 */
void Octree::adjustBoundingBox(ParticleSystem* p){
  float min_x = std::numeric_limits<float>::max();
  float min_y = min_x;
  float min_z = min_x;
  // lowest(), not min(): min() is the smallest positive normal value, which would
  // leave the maximum stuck near zero if every particle sat on the negative side
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = max_x;
  float max_z = max_x;

  // Find the max and min positions
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
}

/**
 * Clears the tree and recomputes the root's bounding box
 * O(n)
 * @param p Particles
 */
void Octree::reset(ParticleSystem* p){
  this->adjustBoundingBox(p);
  Node &root = this->nodes[0];
  root.setFirstChild(-1);
  root.setMass(glm::vec4(-1.f, 0.f, 0.f, 0.f));
  this->nodeCount = 1;
  this->parentCount = 0;
  root.setNext(-1);
}


/**
 * Inserts a particle into the octree (iterative, stackless).
 * Average O(log n) per particle → O(n log n) total.
 *
 * Three cases at each visited node:
 *   1. Empty leaf  — store the particle here directly.
 *   2. Occupied leaf — the node already holds a particle; subdivide into 8
 *      children and re-insert both particles.  If they map to the same child,
 *      keep subdividing until they diverge (or are coincident within 1e-6).
 *   3. Internal node — follow the child that contains pos.
 */
void Octree::insert(glm::vec4 pos, glm::vec4 mass, int root){
  int i = root;

  while(i < this->maxNodes){
    Node &node = this->nodes[i];

    // Case 1: empty leaf — store the particle here
    if(node.isLeaf() && !node.isOccupied()){
      this->insertParticle(pos, mass, i);
      return;
    }

    // Case 2: occupied leaf — need to subdivide and re-insert both particles
    if(node.isLeaf() && node.isOccupied()){
      glm::vec4 pos2 = node.getCenterOfMass();
      glm::vec4 mass2 = glm::vec4(node.getMass(), 0.f, 0.f, 0.f);

      this->subdivide(i);

      int childIndex1 = this->getNextNode(pos, i);
      int childIndex2 = this->getNextNode(pos2, i);
      const float dist = glm::length2(pos2-pos);

      // Both particles in the same octant — keep subdividing until they separate
      while (childIndex1 == childIndex2) {
        // Coincident particles: merge mass into one node to avoid infinite recursion
        if (dist < 1e-6) {
          this->insertParticle(pos, glm::vec4(mass2.x + mass.x, 0.f, 0.f, 0.f), childIndex1);
          return;
        }

        this->subdivide(childIndex1);

        childIndex1 = this->getNextNode(pos, childIndex1);
        childIndex2 = this->getNextNode(pos2, childIndex2);
      }

      this->insertParticle(pos, mass, childIndex1);
      this->insertParticle(pos2, mass2, childIndex2);

      return;
    }

    // Case 3: internal node — descend into the appropriate child
    i = this->getNextNode(pos, i);
  }

}


/**
 * Returns the index of the child node whose octant contains pos.
 * The octant is encoded as a 3-bit index: bit0=x, bit1=y, bit2=z,
 * where bit=1 means the coordinate is in the lower half.
 */
int Octree::getNextNode(glm::vec4 pos, int i){
  Node &node = this->nodes[i];
  const glm::vec4 quadCenter = node.getQuadrantCenter();
  const int quadrantIndex = (pos.x < quadCenter.x) | (pos.y < quadCenter.y) << 1 | (pos.z < quadCenter.z) << 2;
  return node.getFirstChild() + quadrantIndex;
}

/**
 * Creates 8 new empty leaf nodes for the given node
 * @param i Node index
 */
void Octree::subdivide(int i){
  Node &node = this->nodes[i];

  // Set the first child to the current node count
  node.setFirstChild(this->nodeCount);

  // Mark this node as a parent
  this->parents[this->parentCount] = i;
  this->parentCount+=1;

  // Create the next child nodes
  this->nodeCount += 8;

  const glm::vec4 maxB = node.getMaxBoundary();
  const glm::vec4 minB = node.getMinBoundary();

  const float width = maxB.x - minB.x;
  const float height = maxB.y - minB.y;
  const float depth = maxB.z - minB.z;

  int currentChild = node.getFirstChild();
  int firstChild = currentChild;

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
  this->nodes[firstChild + 7].setNext(node.getNext());
}


/**
 * Once the particles are all inserted, propagate (bottom-up) the masses and centers of mass
 * O(p*8) where p is the number of parent nodes
 */
void Octree::propagate() {

  for (int i = parentCount - 1; i >= 0; i--) {
    // Compute the center of mass and mass of the current parent node
    int parentIndex = parents[i];
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
 * Removes the empty nodes from the tree.
 * O(p*8) where p is the number of parent nodes
 */
void Octree::prune() {
  for (int i = 0; i < parentCount; i++) {
    int parentIndex = parents[i];
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

    this->nodes[lastChild].setNext(this->nodes[parentIndex].getNext());
    parent.setFirstChild(firstChild);
  }
}

/**
 * Computes the net gravitational force on a single particle using the
 * Barnes-Hut approximation with a stackless linked-list tree walk.
 *
 * At each node, the opening criterion s^2 < theta * d^2 decides whether
 * to treat the whole subtree as a single body (approximate) or to descend.
 * theta is stored pre-squared (0.25 = 0.5^2) so only a multiply is needed.
 */
glm::vec4 Octree::computeGravityForce(glm::vec4& pos, const float squaredSoftening, const float G) {
  glm::vec4 force = glm::vec4(0.f);

  int i = 0; // start at root

  while (i >= 0){
    Node &node = this->nodes[i];
    glm::vec4 size = node.getMaxBoundary() - node.getMinBoundary();
    float s = glm::max(glm::max(size.x, size.y), size.z); // longest side
    float s_squared = s*s;

    glm::vec4 centerOfMass = node.getCenterOfMass();
    const glm::vec4 vector_i_j = centerOfMass - pos;
    const float dist_squared = glm::length2(vector_i_j);

    // Approximate if leaf or node looks point-like from this particle's position
    if (node.isLeaf() || s_squared < this->theta * dist_squared) {
      if (node.isOccupied() && dist_squared > 0) {
        const float effective_dist_squared = dist_squared + squaredSoftening;
        const float inv_dist = 1.0f / std::pow(effective_dist_squared, 1.5f);
        force += ((vector_i_j * (G * node.getMass())) * inv_dist);
      }

      // Stackless: jump to pre-linked next sibling/ancestor
      i = node.getNext();
    }
    else {
      i = node.getFirstChild(); // descend
    }
  }

  return force;
}

/**
 * Inserts the particle into the node i
 * @param centerOfMass
 * @param mass
 * @param i
 */
void Octree::insertParticle(glm::vec4 centerOfMass, glm::vec4 mass, int i){
  Node &node = this->nodes[i];
  node.setCenterOfMass(centerOfMass);
  node.setMass(mass);
}

Node* Octree::getNodes() {
  return this->nodes;
}

int Octree::getMaxNodes() {
  return this->maxNodes;
}

Octree::~Octree(){
  //delete[] this->nodes;
  delete[] this->parents;
}

int Octree::getNodeCount() {
  return nodeCount;
}

void Octree::setNodes(Node *newNodes) {
  delete[] this->nodes;
  this->nodes = newNodes;
}
