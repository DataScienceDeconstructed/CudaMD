#include "Kernels.cuh"

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

__global__
void calc_v2(float3* v, float* rValues, int threadMax, int SystemParticles){
  //calculates the dot product of a particle's velocity with itself
  int index = threadMax*blockIdx.x + threadIdx.x;
  if(index < SystemParticles){
    rValues[index] = v[index].x * v[index].x + v[index].y * v[index].y + v[index].z * v[index].z ;
  }
}

__global__
void assign_particles2box(float3* r, int* BlockIndex, float3 BoxLengths, int3 BoxNumsXYZ, int threadMax, int SystemParticles){
  /*
  function figures out which box a particle is in
  the indexing happens with a grid that only has blocks in the x direction and threads in the x direction to maximize
  Args:
    r: location of particles in float 3 format
    BlockIndex: array to hold results
    BlockLengths: the physical dimensions of a single block
    BlocksNumXYZ: the number of blocks in each direction in the grid
    threadMax: maximum number of threads in a block
    
  */
  int particleindex = threadMax*blockIdx.x + threadIdx.x;
  if (particleindex<SystemParticles){
    int3 blockindexes;
    blockindexes.x = r[particleindex].x / BoxLengths.x;
    blockindexes.y = r[particleindex].y / BoxLengths.y;
    blockindexes.z = r[particleindex].z / BoxLengths.z;
  
    int particlebox = blockindexes.x + BoxNumsXYZ.x * blockindexes.y + BoxNumsXYZ.x*BoxNumsXYZ.y*blockindexes.z;
  
    BlockIndex[particleindex] = particlebox ;
  }
}

__global__
void assign_box_particles(int* BlockIndex, int threadMax, int SystemParticles, int BoxNumTotal, int MaxPaticles_Box, int* BoxParticles){
  
  //get current thread's box index
  int index = threadMax*blockIdx.x + threadIdx.x;
  //check to see that there is a box that equates to the current thread
  if (index < BoxNumTotal){
    //got through the full array of box indices
    int particleCount = 0; 
    for(int i =0; i<SystemParticles; i++){
      
      //record the ones that belong to this thread's box
      if(BlockIndex[i] == index){
        BoxParticles[particleCount + MaxPaticles_Box*index] = i;
        particleCount++;

      }

    }

    //flag unused portion of the BoxParticles array
    for(int i = particleCount;i<MaxPaticles_Box;i++){
      BoxParticles[particleCount + MaxPaticles_Box*index] = -1;
    }
  }
}