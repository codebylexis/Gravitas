#version 430 core

// Barnes-Hut force evaluation — GPU compute shader
//
// Each invocation handles one particle.  The octree was built on the GPU
// (expandOctree, propagateFatherOctree, etc.) and lives in nodesBuffer.
//
// Traversal (stackless linked-list walk):
//   Start at root (i=0). At each node:
//     - Compute s (longest bounding-box side) and d^2 (distance to particle).
//     - Approximate if leaf OR s^2 < theta * d^2  (node looks point-like).
//       Then jump to node.next (pre-linked sibling/ancestor — no stack needed).
//     - Otherwise descend into node.firstChild.
//
// Optimisation: the first BLOCK_SIZE nodes (upper tree levels, "father nodes")
// are loaded into shared memory once per work-group so all 1024 threads hit
// the same fast cache instead of global memory on every visit.

#define BLOCK_SIZE 1024

layout( local_size_x = BLOCK_SIZE, local_size_y =1, local_size_z = 1  ) in;

uniform float squaredSoftening; // epsilon^2, softens force at r→0 (Plummer softening)
uniform float G;                // gravitational constant (1.0 in simulation units)
uniform int numParticles;
uniform float theta;            // Barnes-Hut criterion stored as theta^2; smaller = more accurate
uniform int fatherTreeNodes;    // number of upper-level nodes cached in shared memory

struct Node{
    vec4 mass; // mass = mass.x; firstChild=mass.y; next=mass.z; maxBoundary.x=mass.w
    vec4 centerOfMass; // (x1*m1 + x2*m2) / (m1 + m2); maxBoundary.y = centerOfMass.w
    vec4 minBoundary;  // maxBoundary.z = minBoundary.w
};

layout(std430, binding=0) buffer positionsBuffer
{
    vec4 positions[];
};


layout(std430, binding=4) buffer forcesBuffer
{
    vec4 forces[];
};

layout(std430, binding=5) buffer nodesBuffer
{
    Node nodes[];
};


bool isLeaf(float firstChild){
    return firstChild < 0.f;
}

bool isOccupied(float mass) {
    return mass > 0.f;
}
// Upper tree levels cached here; all threads in the work-group share these reads
shared Node sharedNodes[BLOCK_SIZE];

void loadSharedNodes(uint threadId){
    if(threadId < BLOCK_SIZE && threadId < fatherTreeNodes){
        sharedNodes[threadId] = nodes[threadId];
    }
    groupMemoryBarrier(); // ensure all writes to shared memory are visible
    barrier();            // ensure all threads have finished loading before traversal begins
}


void main() {
    uint index = gl_GlobalInvocationID.x;
    
    loadSharedNodes(gl_LocalInvocationID.x);

    if (index < numParticles) {



        vec3 force = vec3(0.f);

        int i = 0; // Root of the tree

        Node node; 
        while (i >= 0){
        
            // Get node i
            node = i < BLOCK_SIZE && i < fatherTreeNodes  ? sharedNodes[i] : nodes[i];
            

            const vec4 mass = node.mass;
            const vec4 centerOfMass = node.centerOfMass;
            const vec4 minBoundary = node.minBoundary;
            // maxBoundary was packed into the w-channels of the other vec4s (see Node layout)
            const vec3 maxBoundary = vec3(mass.w, centerOfMass.w, minBoundary.w);

            const vec3 size = maxBoundary - minBoundary.xyz;
            const float s = max(max(size.x, size.y), size.z); // longest side of bounding box
            const float s_squared = s*s;

            const vec3 vector_i_j = centerOfMass.xyz - positions[index].xyz;
            const float dist_squared = dot(vector_i_j, vector_i_j);

            const float firstChild = mass.y;
            const float next = mass.z;

            // Approximate when the node appears point-like: s/d < sqrt(theta)  ⟺  s^2 < theta*d^2
            if (isLeaf(firstChild) || s_squared < theta * dist_squared) {

                if (isOccupied(mass.x) && dist_squared > 0.f) {
                    const float effective_dist_squared = dist_squared + squaredSoftening;
                    // inv_dist = 1/r^3 for F = G*M*vec/r^3
                    const float inv_sqrt_val = inversesqrt(effective_dist_squared);
                    const float inv_dist = inv_sqrt_val * inv_sqrt_val * inv_sqrt_val;

                    force += ((vector_i_j * (G * mass.x)) * inv_dist);
                }

                // Stackless: the "next" pointer was set during tree construction to skip
                // this node's subtree entirely (points to next sibling or ancestor)
                i = int(next);
            }
            else {
                i = int(firstChild); // descend into children
            }
        }



        forces[index] = vec4(force, 0.f);

    }

}


