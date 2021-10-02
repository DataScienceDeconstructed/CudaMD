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
    //go through the full array of box indices
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

__global__
void calculate_accelerations(float3* r, int* BlockIndex, int threadMax, int BoxNumTotal, int* BoxParticles, int MaxPaticles_Box){
//get current thread's box index
  int index = threadMax*blockIdx.x + threadIdx.x;
  
  //check to see that there is a box that equates to the current thread
  if (index < BoxNumTotal){
    //cycle through the particles in this box
    for(int i = 0;i<MaxPaticles_Box;i++){
      //only process if the particle index is nonnegative
      if (BoxParticles[i + MaxPaticles_Box*index] == -1){
        continue;
      }
      
      for(int j = i+1;j<MaxPaticles_Box;j++){
        //only process if the particle index is nonnegative
        if (BoxParticles[j + MaxPaticles_Box*index] == -1){
          continue;
        }else{
        // there are at least 2 particles in this box that will interact.
        // TODO add interaction calculations:

        }

      }

    
    }

  }

}

//TODO test box NearestNeighbors calculation
__global__ 
void calculateNearestNeighbors(int3 VolumeBoxDims, int threadMax, int BoxNumTotal, int4 * CurrentPlaneNeighbors,
int3 * SubordinatePlaneForeNeighbors, int3 * SubordinatePlaneZXPlaneNeighbors, int3 * SubordinatePlaneAftNeighbors){
  
  /* all of the neighbor arrays are being used to house nearest neighbors because they are convient and logically
  seperate the nearest neighbor blocks. the w,x,y,and z coordinates for these varaibles are not meant to indicate
  a physical relationship. They are just used for housing values in a repeatable way
  */
  
  //get current thread's box index
  int index = threadMax*blockIdx.x + threadIdx.x;
  
  //check to see that there is a box that equates to the current thread
  if (index < BoxNumTotal){
  
  
  
  
  int3 boxCoords;
  boxCoords.x = index % VolumeBoxDims.x;
  boxCoords.y = index / VolumeBoxDims.y;
  boxCoords.z = index / VolumeBoxDims.x*VolumeBoxDims.y;

  int3 neighCoords; // dummy variable for holding the coordinates of the neighbor being calculated currently
  
  // look at the box's current plane
  // neighbor 1
  neighCoords.x = (boxCoords.x - 1 ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y + 1 ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z;                          // same plane

  CurrentPlaneNeighbors[index].w = neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;
  
  // neighbor 2
  neighCoords.x = (boxCoords.x     ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y + 1 ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z;                          // same plane

  CurrentPlaneNeighbors[index].x = neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;

  // neighbor 3
  neighCoords.x = (boxCoords.x + 1 ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y + 1 ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z;                          // same plane

  CurrentPlaneNeighbors[index].y = neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;

  // neighbor 4
  neighCoords.x = (boxCoords.x + 1 ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y     ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z;                          // same plane

  CurrentPlaneNeighbors[index].z = neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;

  // look at the subordinate plane
  
  //Fore neighbors
  // neighbor 5
  neighCoords.x = (boxCoords.x - 1 ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y + 1 ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z - 1;                      // lower plane
  
  // no PBC on the Z axis
  SubordinatePlaneForeNeighbors[index].x = (neighCoords.z == -1) ? -1 : neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;
  
  // neighbor 6
  neighCoords.x = (boxCoords.x     ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y + 1 ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z - 1;                      // lower plane

  //no PBC on the Z axis
  SubordinatePlaneForeNeighbors[index].y = (neighCoords.z == -1) ? -1 : neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;

  // neighbor 7
  neighCoords.x = (boxCoords.x + 1 ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y + 1 ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z -1 ;                      // lower plane

  //no PBC on the Z axis
  SubordinatePlaneForeNeighbors[index].z = (neighCoords.z == -1) ? -1 : neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;

  //ZX Plane neighbors
  // neighbor 8
  neighCoords.x = (boxCoords.x - 1 ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y     ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z - 1;                      // lower plane
  
  // no PBC on the Z axis
  SubordinatePlaneZXPlaneNeighbors[index].x = (neighCoords.z == -1) ? -1 : neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;
  
  // neighbor 9
  neighCoords.x = (boxCoords.x     ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y     ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z - 1;                      // lower plane

  //no PBC on the Z axis
  SubordinatePlaneZXPlaneNeighbors[index].y = (neighCoords.z == -1) ? -1 : neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;

  // neighbor 10
  neighCoords.x = (boxCoords.x + 1 ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y     ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z -1 ;                      // lower plane

  //no PBC on the Z axis
  SubordinatePlaneZXPlaneNeighbors[index].z = (neighCoords.z == -1) ? -1 : neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;

  //aft neighbors
  // neighbor 11
  neighCoords.x = (boxCoords.x - 1 ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y - 1 ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z - 1;                      // lower plane
  
  // no PBC on the Z axis
  SubordinatePlaneAftNeighbors[index].x = (neighCoords.z == -1) ? -1 : neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;
  
  // neighbor 12
  neighCoords.x = (boxCoords.x     ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y - 1 ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z - 1;                      // lower plane

  //no PBC on the Z axis
  SubordinatePlaneAftNeighbors[index].y = (neighCoords.z == -1) ? -1 : neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;

  // neighbor 13
  neighCoords.x = (boxCoords.x + 1 ) % VolumeBoxDims.x; // ensure Periodic boundary conditions
  neighCoords.y = (boxCoords.y - 1 ) % VolumeBoxDims.y; // ensure Periodic boundary conditions
  neighCoords.z = boxCoords.z -1 ;                      // lower plane

  //no PBC on the Z axis
  SubordinatePlaneAftNeighbors[index].z = (neighCoords.z == -1) ? -1 : neighCoords.x + neighCoords.y * VolumeBoxDims.x + neighCoords.z * VolumeBoxDims.x * VolumeBoxDims.y ;

  }
}