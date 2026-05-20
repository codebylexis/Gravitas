#ifndef NODE_H
#define NODE_H

#include <glm/glm.hpp>
#include <ostream>

// A single node in the Barnes-Hut octree.
//
// To keep the struct GPU-friendly (std430 layout), all data is packed into
// three vec4s rather than using separate fields.  The layout is:
//
//   mass.x         — total mass (negative means empty/unoccupied)
//   mass.y         — firstChild index (negative means this is a leaf)
//   mass.z         — next sibling/ancestor index for the stackless traversal
//   mass.w         — maxBoundary.x
//   centerOfMass.x — center-of-mass x
//   centerOfMass.y — center-of-mass y
//   centerOfMass.z — center-of-mass z
//   centerOfMass.w — maxBoundary.y
//   minBoundary.x  — minBoundary.x
//   minBoundary.y  — minBoundary.y
//   minBoundary.z  — minBoundary.z
//   minBoundary.w  — maxBoundary.z
//
// The "next" pointer enables a stackless tree walk: after processing a leaf
// (or an approximated internal node), jump to next instead of popping a stack.
struct alignas(16) Node {
  public:
    glm::vec4 mass;         // mass.x=mass, mass.y=firstChild, mass.z=next, mass.w=maxBound.x
    glm::vec4 centerOfMass; // xyz=CoM, w=maxBound.y
    glm::vec4 minBoundary;  // xyz=minBound, w=maxBound.z

    Node();
    ~Node();

    void setFirstChild(int firstChild){
        this->mass.y = static_cast<float>(firstChild);
    }

    inline int getFirstChild() const{
        return static_cast<int>(this->mass.y);
    }

    void setNext(int next) {
        this->mass.z = static_cast<float>(next);
    }

    inline int getNext() const{
        return static_cast<int>(this->mass.z);
    }

    void setMass(glm::vec4 mass){
        this->mass.x = mass.x;
    }

    void setCenterOfMass(glm::vec4 centerOfMass){
        this->centerOfMass.x = centerOfMass.x;
        this->centerOfMass.y = centerOfMass.y;
        this->centerOfMass.z = centerOfMass.z;
    }

    inline void setBoundingBox(glm::vec4 minBoundary, glm::vec4 maxBoundary){
        this->minBoundary.x = minBoundary.x;
        this->minBoundary.y = minBoundary.y;
        this->minBoundary.z = minBoundary.z;
        // maxBoundary components are packed into the w-channels of the other vec4s
        this->mass.w = maxBoundary.x;
        this->centerOfMass.w = maxBoundary.y;
        this->minBoundary.w = maxBoundary.z;
    }

    inline void createEmptyNode(glm::vec4 minBoundary, glm::vec4 maxBoundary) {
        this->mass.x = -1.f;  // mark unoccupied
        this->setFirstChild(-1);
        this->setBoundingBox(minBoundary, maxBoundary);
    }

    inline glm::vec4 getMinBoundary() const{
        glm::vec4 v = this->minBoundary;
        return {v.x, v.y, v.z, 0.f};
    }
    inline glm::vec4 getMaxBoundary() const{
        // maxBoundary was packed across the w-channels; reconstruct it here
        return {mass.w, centerOfMass.w, minBoundary.w, 0.f};
    }

    inline glm::vec4 getCenterOfMass() const {
        glm::vec4 c = centerOfMass;
        return {c.x, c.y, c.z, 0.f};
    }

    inline float getMass() const{
        return this->mass.x;
    }

    // A node is a leaf when it has no children (firstChild < 0)
    inline bool isLeaf() const{
        return getFirstChild() < 0;
    }

    // A node is occupied when it holds at least one particle (mass > 0)
    inline bool isOccupied() const{
        return this->mass.x > 0.f;
    }

    // Returns the center point of this node's bounding box (used to pick the child octant)
    inline glm::vec4 getQuadrantCenter() const {
        glm::vec4 maxBoundary = getMaxBoundary();
        glm::vec4 minBoundary = getMinBoundary();
        const float width = maxBoundary.x - minBoundary.x;
        const float height = maxBoundary.y - minBoundary.y;
        const float depth = maxBoundary.z - minBoundary.z;
        return {minBoundary.x + (width / 2.f), minBoundary.y + (height / 2.f), minBoundary.z + (depth / 2.f), 0.f};
    }
};


#endif //NODE_H
