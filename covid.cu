#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h> 
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
typedef struct
{
	int x, y, time_infected, R, state;
	bool symptoms;
} person;

typedef enum {None, Avoid, Central} behavior;

typedef struct
{
	person *list;
} people;

// all of the people
person *g_population=NULL;

// people stored by location
//person ***g_world=NULL;

// Current width of world.
size_t g_worldWidth=0;

/// Current height of world.
size_t g_worldHeight=0;

size_t g_popSize = 0;

static inline void covid_initMaster(unsigned int pop_size, size_t world_width, size_t world_height, curandState *state) {
	g_worldWidth = world_width;
	g_worldHeight = world_height;
	g_popSize = pop_size;


	// 
	int world_area = g_worldHeight * g_worldWidth;
	float pop_density = g_popSize / world_area;
	int depth = ceil(pop_density)*5;


	cudaMallocManaged((void**)&g_population, g_popSize*sizeof(person));
	//person *world;
	//cudaMallocManaged((void**)world, g_worldHeight*g_worldWidth*depth*sizeof(person));
	//g_world = (person (**)[depth]) world;

	for (int i=0; i<g_popSize; ++i) {
		int x = rand() % (g_worldWidth);
		int y = rand() % (g_worldHeight);

		g_population[i].x = x;
		g_population[i].y = y;
		g_population[i].time_infected = -1;
		g_population[i].R = 0;
		g_population[i].state = 0;
		g_population[i].symptoms = false;
	}

}

__device__
int compare(const void* a, const void* b) {
    // sort into bins based on x+y
    person p1 = * ( (person*) a);
    person p2 = * ( (person*) b);
    float p1_dist, p2_dist;

    if (p1.x == p2.x && p1.y == p2.y) return 0; // this shouldn't happen with no collisions allowed
    
    p1_dist = p1.x+p1.y;
    p2_dist = p2.x+p2.y;

    if (p1_dist == p2_dist) return 0;
    else if (p1_dist < p2_dist) return -1;
    else return 1;

}
/*
__device__
int max(int a, int b) {
    return (a > b) ? a : b;
}

__device__
int min(int a, int b) {
    return (a < b) ? a : b;
}
*/
__device__
int get_curand(curandState *state, int index, int min, int max) {
    https://stackoverflow.com/a/18501534/13254229
    float myrandf = curand_uniform(state+index);
    myrandf *= (max - min+0.999999);
    myrandf += min;
    return (int)truncf(myrandf);
}

__device__
int roll(float chance, curandState *state, int index) {
    return get_curand(state, index, 0, 100) <= chance*100;
}

__device__
void noRestraints( int x, int y, int time, int *new_x, int *new_y ){
	//in which a pseudorandom location is chosen as the next 

	int i = (x*y+1-time)%7;

	switch (i) {
        case 0:
            *new_x = x+time%3-1;
            *new_y = y-1;
            break;
        case 1:
            *new_x = x+1;
            *new_y = y+1;
            break;
        case 2:
            *new_x = x+time%3-1;
            *new_y = y-1;
            break;
        case 3:
            *new_x = x-1;
            *new_y = y+time%3-1;
            break;
        case 4:
            *new_x = x-1;
            *new_y = y;
            break;
        case 6:
            *new_x = x+time%3-1;
            *new_y = y;
            break;
        default:
            *new_x = x+time%3-1;
            *new_y = y-time%3+1;
            break;
    }

}

__device__ 
void avoid(int x, int y, int near_x, int near_y, int *new_x, int *new_y){
	int leftright = x-near_x;
	int updown = y-near_y;

	if(leftright>1){
		*new_x = x+1;
	}
	else{
		*new_x = x-1;
	}

	if(updown>1){
		*new_y = y+1;
	}
	else{
		*new_y = y-1;
	}

}

 __device__
void central(int x, int y, int cen_x, int cen_y, int time, int *new_x, int *new_y){
	int leftright = x-cen_x;
	int updown = y-cen_y;

	if (leftright*leftright+updown*updown <20){
		//chance to leave
		if(leftright+time*updown%6 == 2){
			*new_x = x + 30*(time+updown)%5 - 30;
			*new_y = y - 30*(time+updown)%5 + 30;
		}
		else{
			noRestraints(x, y, time, new_x, new_y);
		}
	}
	else{
		if(leftright+time*updown%6 == 2){
			*new_x = cen_x+time%3-1;
			*new_y = cen_x-(time+leftright)%3+1;
		}
		else{
			noRestraints(x, y, time, new_x, new_y);
		}

	}

}
__global__
void covid_kernel(
    const person* d_population, 
    person* d_result, 
    unsigned int world_width, 
    unsigned int world_height, 
    int time, 
    unsigned int pop_size, 
    unsigned int radius, 
    float infect_chance, 
    float symptom_chance, 
    int infect_search, 
    unsigned int recover, 
    int threshold,
    int behavior1,
    int behavior2, 
    int myrank, 
    int numranks,
    curandState *d_state
) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int start_i, end_i, i, infected_count = 0, behavior, nearest_search, new_x, new_y;
    float dist, nearest_dist;
    person *nearest = NULL;
    while (index < pop_size) {
        // if infected, check within radius to spread infection
        if (d_population[index].state == 1) {
            start_i = max(index-infect_search, 0);
            end_i = min(index+infect_search, pop_size-1);
            for (i = start_i; i < end_i; ++i) {
                if (d_population[i].state == 0 && abs(d_population[i].x - d_population[index].x) <= radius && abs(d_population[i].y - d_population[index].y) <= radius) {
                    if (roll(infect_chance, d_state, index)) {
                        d_result[i].state = 1;
                        d_result[i].time_infected = time;
                        d_result[i].symptoms = roll(symptom_chance, d_state, index);
                        d_result[index].R++;
                    }
                }
            }
            // if no symptoms, roll for chance to start showing symptoms
            if (!d_population[index].symptoms) {
                d_result[i].symptoms = roll(symptom_chance, d_state, index);
            } else { // if showing symptoms add to ifected count
                infected_count++;
            }
            // check if person recovers
            if (d_population[index].time_infected + recover >= time) {
                d_result[index].state = 2;
            }
            // maybe add death infect_chance?
        }
        // check global status to decide behavior
        if (infected_count > threshold) {
            behavior = behavior2;    
        } else {
            behavior = behavior1;
        }

        switch (behavior) {
            case None: 
                noRestraints(d_population[index].x, d_population[index].y, time, &new_x, &new_y);
                d_result[index].x = new_x; 
                d_result[index].y = new_y;
                break;
            case Avoid:
                // find nearest
                nearest_search = (int)(max(world_width, world_height)*3/2)+1;
                start_i = max(index-nearest_search, 0);
                end_i = min(index+nearest_search, pop_size-1);
                for (i=start_i; i < end_i; ++i) {
                    if (!nearest) {
                        *nearest = d_population[i];
                        nearest_dist = sqrt(pow(d_population[i].x- d_population[index].x, 2)+pow(d_population[i].y-d_population[index].y, 2));
                    }
                    dist = sqrt(pow(d_population[i].x-d_population[index].x, 2)+pow(d_population[i].y-d_population[index].y, 2));
                    if (dist < nearest_dist) {
                        *nearest = d_population[i];
                        nearest_dist = dist;
                    }
                }
                avoid(d_population[index].x, d_population[index].y, (*nearest).x, (*nearest).y, &new_x, &new_y); 
                d_result[index].x = new_x; 
                d_result[index].y = new_y;
                break;
            case Central:
                central(d_population[index].x, d_population[index].y, (int)world_width/2, (int)world_height/2, time, &new_x, &new_y); 
                d_result[index].x = new_x; 
                d_result[index].y = new_y;
                break;
            default:
                break;
        }

        index += blockDim.x * gridDim.x;
    }
}

__global__ void setup_kernel(curandState *state){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

bool covid_kernelLaunch(
    person** d_population, 
    person** d_result, 
    size_t world_width, 
    size_t world_height, 
    size_t pop_size, 
    int time, 
    unsigned int radius, 
    float infect_chance, 
    float symptom_chance, 
    unsigned int recover, 
    int threshold, 
    int behavior1, 
    int behavior2, 
    int myrank, 
    int numranks
) {
    int dim_max = max(world_width, world_height);
    int infect_search = ceil((2*radius + 1) * dim_max / 2);
    curandState *d_state;
    qsort(*d_population, pop_size, sizeof(person), compare);
    setup_kernel<<<1,1>>>(d_state);
    covid_kernel<<<1,1>>>(*d_population, *d_result, (int)world_width, (int)world_height, time, pop_size, radius, infect_chance, symptom_chance, infect_search, recover, threshold, behavior1, behavior2, myrank, numranks, d_state);
    cudaDeviceSynchronize();
    return true;
}
/*
int main(int argc, char *argv[]) {
	unsigned int pop_size, world_width, world_height, infection_radius, infection_infect_chance, days;

	pop_size = atoi(argv[1]);
	world_width = atoi(argv[2]);
	world_height = atoi(argv[3]);
	infection_radius = atoi(argv[4]);
	days = atoi (argv[4]);
	unsigned int timesteps = days*24;

	srand(time(0));
	covid_initMaster(pop_size, world_width, world_height);

    for (int i=0; i < g_popSize; ++i) {
        printf("%d, %d\n", g_population[i].x, g_population[i].y);
    }
}
*/

