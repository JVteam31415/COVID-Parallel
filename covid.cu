#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include<time.h> 
#include<math.h>

typedef struct
{
	int x, y, day_infected, R, state;
	bool symptoms;
} person;

typedef struct
{
	person *list;
} people;

// all of the people
person *g_population=NULL;

// people stored by location
person ***g_world=NULL;

// Current width of world.
size_t g_worldWidth=0;

/// Current height of world.
size_t g_worldHeight=0;

size_t g_popSize = 0;

static inline void covid_initMaster(unsigned int pop_size, size_t world_width, size_t world_height) {
	g_worldWidth = world_width;
	g_worldHeight = world_height;
	g_popSize = pop_size;


	// 
	int world_area = g_worldHeight * g_worldWidth;
	float pop_density = g_popsize / world_area;
	int depth = ceil(pop_density)*5;


	cudaMallocManaged((void**)&g_population, g_popSize*sizeof(person));
	person *world;
	cudaMallocManaged((void**)world, g_worldHeight*g_worldWidth*depth*sizeof(person));
	g_world = (person (**)[depth]) world;



	for (int i <)
}

int main(int argc, char *argv[]) {
	unsigned int pop_size, world_width, world_height, infection_radius, infection_chance;

	srand(time(0));
	covid_initMaster(pop_size, world_width, world_height);


}
