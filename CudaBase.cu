#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>     /* srand, rand */

#include "Kernels.cuh"


int writePositions(float3 * r, int size);

int main(void)
{
  //***************************
  //define Simulation Variables
  //***************************

  //define number of each type of object in the simulation
  int NanoCount = 10;            // number of nanoparticles  in the system
  int PolymerCount = 10;          // number of Polymer chains in the system

  //length related variables
  float R = 1.0;                  // Radius of a single monomer
  float Rc = 3.0*R;               // cutoff radius for interactions
  int PolymerMaxLength = 10;      // Max number of Monomers in a Polymer Chain
  float PolymerSigma = .1;        // Global Density of Polymer chains attached to the volume floor: Units = Number of Polymer / Rc^2

  //derived scale values
  float SubstrateSurfaceArea = ((float)PolymerCount)/ PolymerSigma;               // just rearranging the relationship sigma = number / area
  float VolumeDepth = ((float) ((int) sqrt(SubstrateSurfaceArea) + 1.0) ) * Rc;   // X-Axis: Depth * Length must equal the Area so we just divide evenly between the two with the square root. 
  float VolumeLength = ((float) ((int) sqrt(SubstrateSurfaceArea) + 1.0) ) * Rc;  // Y-Axis: the plus one and casting craziness is to get a ceiling operation and then convert back to a float.
  float VolumeHeight = (float) PolymerMaxLength * R * 2.0 ;                       // Z-Axis: must be > Polymer length to make sure that the polymers will have room to expand

  // setup polymers and determine total number of System Particles
  int PolymerLengths[PolymerCount];
  int SystemParticles = 0;
  int PolymerLength = 0;
  for(int i = 0;i<PolymerCount;i++){
    PolymerLength = 2 + (i % (PolymerMaxLength-2));
    SystemParticles+= PolymerLength;                    //add the number of monomers in the current polymer chain to the total count
    PolymerLengths[i] = PolymerLength;                  //this should be random, but qualitatively I'm just making the polymers have different lengths
    
  }

  SystemParticles += NanoCount;     //this is now the total number of particles (nano plus monomers) in the system.
  
  //Define Kinematic Data memory
  //float *x, *y;
  //cudaMallocManaged(&x, N*sizeof(float));

  int *Type;
  int2 * neighbors;
  float3 *r, *v, *a;

  cudaMallocManaged(&Type, SystemParticles*sizeof(int));
  cudaMallocManaged(&neighbors, SystemParticles*sizeof(int2));
  cudaMallocManaged(&r, SystemParticles*sizeof(float3));
  cudaMallocManaged(&v, SystemParticles*sizeof(float3));
  cudaMallocManaged(&a, SystemParticles*sizeof(float3));

  //int Type[SystemParticles];        // particle type we use primes to quickly identify the interaction type (nano = 2, mono = 3, anchor = 5)
  //float3 r[SystemParticles];        // location of particle
  //float3 v[SystemParticles];        // velocity of particle
  //float3 a[SystemParticles];        // acceleartion of particle
  //int2 neighbors[SystemParticles];  // index of connected particles neighbors[x] are the 2 neighbors of x. 
  
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

  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 2.0f;
    y[i] = 2.0f;
  }

  std::cout<< x[54];
  std::cout<< y[54];

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
 std::cout<< y[54];
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;

  for (int i = 0; i < N; i++){
    maxError = fmax(maxError, fabs(y[i]-4.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  cudaFree(Type);
  cudaFree(neighbors);
  cudaFree(r);
  cudaFree(v);
  cudaFree(a);
  
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