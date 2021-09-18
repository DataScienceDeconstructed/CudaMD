#include "./MDUtils.cuh"

bool scale_velocity_for_Temp (int SystemParticles, float mass, float kB, float Temp, float* ScalarTemp, float3* v){
    /*
      Function adjusts velocities so that the systen temperature goes from its current arbitrary value to "Temp".
      Args:
        System Particles: number of particles in the simulation
        mass: mass of particles in the system. assumption is that they all have the same mass. if this changes m needs to be an array
        kB: Boltzman's constanct in Joules / Kelvin
        Temp: the scalar value of temperature that the simulation needs to acheive after rescaling
        ScalarTemp: array of scalars representing the dot product of a particle with itself i.e.  v dot v 
      */
    bool rValue = false;
    
    float meanv2 = 0.0; 
    for (int i = 0;i<SystemParticles;i++){
        meanv2 += ScalarTemp[i];
    }
    meanv2 /= SystemParticles;

    float T_initial = 1.0/3.0*mass*meanv2 / kB; //find initial temperature

    float scale = sqrt(Temp/T_initial);         //find scale factor between current and desired temp. Square root is because we scale v which is squared in the equation
    for (int i = 0;i<SystemParticles;i++){
        v[i].x *= scale;
        v[i].y *= scale;
        v[i].z *= scale;
    }
    rValue = true;
    return rValue;
}