__global__ 
void add(int n, float *x, float *y);
__global__
void calc_v2(float3* v, float* rValues, int threadMax, int SystemParticles);
__global__
void assign_particles2box(float3* r, int* BlockIndex, float3 BoxLengths, int3 BoxNumsXYZ, int threadMax, int SystemParticles);
__global__
void assign_box_particles(int* BlockIndex, int threadMax, int SystemParticles, int BoxNumTotal, int MaxPaticles_Box, int* BoxParticles);
__global__ 
void calculateNearestNeighbors(int3 VolumeBoxDims, int threadMax, int BoxNumTotal, int4 * CurrentPlaneNeighbors,
int3 * SubordinatePlaneForeNeighbors, int3 * SubordinatePlaneZXPlaneNeighbors, int3 * SubordinatePlaneAftNeighbors);