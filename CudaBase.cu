#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include "Kernels.cuh"
#include "MDUtils.cuh"


int writePositions(float3 * r, int size);

int main(void)
{
  //##############
  //CUDA realted variables
  //##############

  // note that all 4 of the maxes below must be staisfied.
  int maxThreadsTotal = 1024;
  int maxThreadsX = 1024;
  int maxThreadsY = 1024;
  int maxThreadsZ = 64;   // not a typo that is the current nVidia max
  int maxBlocksXYZ = 65535;  // technically its 2147483647 but that is only for the x direction and insane

  int numBlocks = 0;
  int numBoxBlocks = 0;

  //***************************
  //define Simulation Variables
  //***************************

  //define number of each type of object in the simulation
  int RequestedNanoCount = 32;    // number of nanoparticles in the system will be set so a multiple of 32 so the final number will be +/- 31 of this number
  int NanoCount;
  int PolymerCount = 10;          // number of Polymer chains in the system

  //length related variables
  float R = 1.0;                  // Radius of a single monomer
  float Rc = 3.0*R;               // cutoff radius for interactions
  int PolymerMaxLength = 10;      // Max number of Monomers in a Polymer Chain
  float PolymerSigma = .01;        // Global Density of Polymer chains attached to the volume floor: Units = Number of Polymers / Rc^2

  //derived scale values
  float SubstrateSurfaceArea = ((float)PolymerCount)/ PolymerSigma;               // just rearranging the relationship sigma = number / area
  float VolumeDepth = ((float) ((int) sqrt(SubstrateSurfaceArea) + 1.0) ) * Rc;   // X-Axis: Depth * Length must equal the Area so we just divide evenly between the two with the square root. 
  float VolumeLength = ((float) ((int) sqrt(SubstrateSurfaceArea) + 1.0) ) * Rc;  // Y-Axis: the plus one and casting craziness is to get a ceiling operation and then convert back to a float.
  float VolumeHeight = (float) PolymerMaxLength * R * 3.0;                        // Z-Axis: must be > Polymer length to make sure that the polymers will have room to expand 2R would be the height
                                                                                  // of a fully extended polymer so we give 50% extra
  // define box dimension data
  float3 BoxLengths;
  BoxLengths.x = Rc;
  BoxLengths.y = Rc;
  BoxLengths.z = Rc;
  float BoxVolume = BoxLengths.x*BoxLengths.y*BoxLengths.z;
  
  //define the grid for the boxes 
  // all values are floats so the +1 makes sure that we don't make the system too small
  int3 BoxNumsXYZ;
  BoxNumsXYZ.x = (VolumeDepth / BoxLengths.x) +1; 
  BoxNumsXYZ.y = (VolumeLength / BoxLengths.y) +1;
  BoxNumsXYZ.z = (VolumeHeight / BoxLengths.z) +1;
  int BoxNumTotal =   BoxNumsXYZ.x * BoxNumsXYZ.y * BoxNumsXYZ.z;

  // define the MaxParticles per box on the assumption that the interactions let twice the volume of the particles into the box. 
  // The factor of 2 is a fudge factor.
  int MaxPaticles_Box = BoxVolume / (4.0 / 3.0 * 3.14 * R * R *R) * 2;

  //Energy related values
  float mass = 1.0;                     // Mass in Kilograms
  float Temp = 300;                     // Temperature in Kelvin
  float kB = 1.38064852 * pow(10,-23);  // Boltzman's constant in Joules / Kelvin

  // setup polymers and determine total number of System Particles
  int PolymerLengths[PolymerCount];
  int SystemParticles = 0;
  int PolymerLength = 0;
  for(int i = 0;i<PolymerCount;i++){
    PolymerLength = 2 + (i % (PolymerMaxLength-2));
    SystemParticles+= PolymerLength;                    //add the number of monomers in the current polymer chain to the total count
    PolymerLengths[i] = PolymerLength;                  //this should be random, but qualitatively I'm just making the polymers have different lengths
    
  }

  //set the system to have a number of particles divisible by 32
  int tExtraNanos = (SystemParticles + RequestedNanoCount) % 32;  // Figures out how many particles over a multiple of 32 we are
  NanoCount = RequestedNanoCount - tExtraNanos;                   // Removes the remainder.
  SystemParticles += NanoCount;                                   // This is now the total number of particles (nano plus monomers) in the system.
  
  //Define Data memory

  int *Type, *BlockIndex, *BoxParticles;
  int2 * neighbors;
  float3 *r, *v, *a;
  float *ScalarTemp;
  int4 * CurrentPlaneNeighbors;
  int3 * SubordinatePlaneForeNeighbors, * SubordinatePlaneZXPlaneNeighbors, * SubordinatePlaneAftNeighbors;

  cudaMallocManaged(&Type, SystemParticles*sizeof(int));
  cudaMallocManaged(&BlockIndex, SystemParticles*sizeof(int));
  cudaMallocManaged(&BoxParticles, MaxPaticles_Box*BoxNumTotal*sizeof(int));
  
  //allocate memory for nearest neighbor recordings
  cudaMallocManaged(&CurrentPlaneNeighbors, BoxNumTotal*sizeof(int4));
  cudaMallocManaged(&SubordinatePlaneForeNeighbors, BoxNumTotal*sizeof(int3));
  cudaMallocManaged(&SubordinatePlaneZXPlaneNeighbors, BoxNumTotal*sizeof(int3));
  cudaMallocManaged(&SubordinatePlaneAftNeighbors, BoxNumTotal*sizeof(int3));
  
  cudaMallocManaged(&neighbors, SystemParticles*sizeof(int2));
  cudaMallocManaged(&r, SystemParticles*sizeof(float3));
  cudaMallocManaged(&v, SystemParticles*sizeof(float3));
  cudaMallocManaged(&a, SystemParticles*sizeof(float3));
  cudaMallocManaged(&ScalarTemp, SystemParticles*sizeof(float));
  
  srand(123);                 //TODO: add better random number generator. seed random numbers probably need to add in a more reliable random number generator
  
  //initialize the system particles
  //TODO: all particle velocities will need to be scaled later on to ensure the temperature of the system.
  for(int i = 0;i<NanoCount;i++){
    Type[i] = 2;// 2 is for nano particles
    r[i].x = ((float)(rand() % 100 + 1))*.01*VolumeDepth; // uniform distribution over the volume cube
    r[i].y = ((float)(rand() % 100 + 1))*.01*VolumeLength; 
    r[i].z = ((float)(rand() % 100 + 1))*.01*VolumeHeight; 
    v[i].x = ((float)(rand() % 100 + 1))*.01 - 0.5; //uniform distribution over momentum space cube
    v[i].y = ((float)(rand() % 100 + 1))*.01 - 0.5; 
    v[i].z = ((float)(rand() % 100 + 1))*.01 - 0.5; 
    a[i].x = 0.0; //start without acceleration
    a[i].y = 0.0;
    a[i].z = 0.0;
    neighbors[i].x = -1; // nanoparticles don't have neighbors
    neighbors[i].y = -1;
      
  }
  
  int InitializationIndex = NanoCount; // this is being done because of the way memory layouts will be used.

  //initialize the polymers
  //the locations need some element of randomness to them
  for(int i = 0;i<PolymerCount;i++){
    for(int j = 0;j<PolymerLengths[i];j++){
      if (j == 0){
        //this is an anchor monomer
        Type[InitializationIndex] = 5;
        r[InitializationIndex].x = ((float)(rand() % 100 + 1))*.01*VolumeDepth; // uniform distribution over the volume substrate
        r[InitializationIndex].y = ((float)(rand() % 100 + 1))*.01*VolumeLength; 
        r[InitializationIndex].z = 0.0; 
        v[InitializationIndex].x = ((float)(rand() % 100 + 1))*.01-0.5; //uniform distribution over momentum space cude
        v[InitializationIndex].y = ((float)(rand() % 100 + 1))*.01-0.5; 
        v[InitializationIndex].z = (((float)(rand() % 100 + 1))*.01-0.5)*.01; // any z velocity will be very small 
        a[InitializationIndex].x = 0.0; //start without acceleration
        a[InitializationIndex].y = 0.0;
        a[InitializationIndex].z = 0.0;
        neighbors[InitializationIndex].x = -1; // anchor particles don't have neighbors
        neighbors[InitializationIndex].y = InitializationIndex+1;

      }else{
        //this is a polymer monomer
        Type[InitializationIndex] = 3;
        if(j == PolymerLengths[i] -1){
          //this monomer is the tail
          r[InitializationIndex].x = r[InitializationIndex-1].x; //keep the previous x coordinate
          r[InitializationIndex].y = r[InitializationIndex-1].y; //keep the previous y coordinate
          r[InitializationIndex].z = r[InitializationIndex-1].z + R; //stack this monomer right on top of the previous one
          v[InitializationIndex].x = ((float)(rand() % 100 + 1))*.01-0.5; // uniform distribution over momentum space cude
          v[InitializationIndex].y = ((float)(rand() % 100 + 1))*.01-0.5; 
          v[InitializationIndex].z = ((float)(rand() % 100 + 1))*.01-0.5;  
          a[InitializationIndex].x = 0.0; //start without acceleration
          a[InitializationIndex].y = 0.0;
          a[InitializationIndex].z = 0.0;
          neighbors[InitializationIndex].x = InitializationIndex-1; 
          neighbors[InitializationIndex].y = -1; // tails don't have right side neighbors
        }else{
          //this monomer is in the middle of the chain
          
          r[InitializationIndex].x = r[InitializationIndex-1].x; // keep x and y and extend upwards
          r[InitializationIndex].y = r[InitializationIndex-1].y; 
          r[InitializationIndex].z = r[InitializationIndex-1].z + R; 
          v[InitializationIndex].x = ((float)(rand() % 100 + 1))*.01-0.5; //uniform distribution over momentum space cube
          v[InitializationIndex].y = ((float)(rand() % 100 + 1))*.01-0.5; 
          v[InitializationIndex].z = ((float)(rand() % 100 + 1))*.01-0.5; 
          a[InitializationIndex].x = 0.0; //start without acceleration
          a[InitializationIndex].y = 0.0;
          a[InitializationIndex].z = 0.0;
          neighbors[InitializationIndex].x = InitializationIndex-1; 
          neighbors[InitializationIndex].y = InitializationIndex+1; 
        }
      }
      InitializationIndex++;
    }
  }


 // set blocks for particle based indexing subroutines like identifying box location and velocity scaling
  numBlocks = SystemParticles / maxThreadsTotal +1; // the plus one is because of the integer division we always need at least 1
  if (numBlocks % maxThreadsTotal ==0){
   // if numblocks is divisble by max threads then we have one extra block
   --numBlocks;
  }

 //set blocks for box based indexing subroutines like putting particles in box arrays
  numBoxBlocks = BoxNumTotal/maxThreadsTotal +1;
  if (numBoxBlocks % maxThreadsTotal ==0){
   // if numblocks is divisble by max threads then we have one extra block
   --numBoxBlocks;
  }

 //we need to scale velocities to confine temperature
 // cuda part of velocity scaling
 dim3 blocks(numBlocks);
 dim3 threads(maxThreadsTotal);
 dim3 boxBlocks(numBoxBlocks);
 
 calc_v2<<<blocks, threads>>>(v, ScalarTemp, maxThreadsTotal,SystemParticles); // TODO CHECK WHAT happens when the array is less than 1024 
 cudaDeviceSynchronize(); // Wait for GPU to finish before accessing on host
 // cpu part of velosity scaling
 if (!scale_velocity_for_Temp ( SystemParticles, mass, kB, Temp, ScalarTemp, v)){
  return -1;
 }

 // assign all particles to boxes
assign_particles2box<<<blocks, threads>>>(r, BlockIndex, BoxLengths, BoxNumsXYZ, maxThreadsTotal, SystemParticles);
cudaDeviceSynchronize();

//construct particle -> box mappings
assign_box_particles<<<boxBlocks, threads>>>(BlockIndex, maxThreadsTotal,  SystemParticles,  BoxNumTotal,  MaxPaticles_Box, BoxParticles);
cudaDeviceSynchronize();
 
//establish nearest neighbors
calculateNearestNeighbors<<<boxBlocks, threads>>>( BoxNumsXYZ, maxThreadsTotal, BoxNumTotal, CurrentPlaneNeighbors,
                          SubordinatePlaneForeNeighbors, SubordinatePlaneZXPlaneNeighbors,  SubordinatePlaneAftNeighbors);
cudaDeviceSynchronize();

  //write out initial configuration
  //writePositions(r,SystemParticles);
  std::cout<<"testing";
  std::ofstream myfile;
  myfile.open ("/home/clayton/Disertation/CudaMD/Particles.csv", std::ofstream::out);
  myfile << "X,Y,Z \n ";

    for(int i = 0;i<SystemParticles;i++){
        myfile << r[i].x << " , " << r[i].y << " , "<< r[i].z << " \n ";
      }
  
  myfile.close();

  
  
  cudaFree(BlockIndex);
  cudaFree(BoxParticles);

  //box nearest neighbor processing
  cudaFree(CurrentPlaneNeighbors);
  cudaFree(SubordinatePlaneAftNeighbors);
  cudaFree(SubordinatePlaneForeNeighbors);
  cudaFree(SubordinatePlaneZXPlaneNeighbors);
  
  //particle related values
  cudaFree(Type);
  cudaFree(neighbors);
  cudaFree(r);
  cudaFree(v);
  cudaFree(a);
  cudaFree(ScalarTemp);
  
  return 0;
}

int writePositions(float3 * r, int size){
      std::ofstream myfile;
      myfile.open ("/home/clayton/Disertation/CudaMD/ParticleLocations.csv",std::ofstream::out);
      for(int i = 0;i<size;i++){
        myfile << r[i].x ,",", r[i].y,",", r[i].z,"\n";
      }
      myfile.close();
  return 0;
}