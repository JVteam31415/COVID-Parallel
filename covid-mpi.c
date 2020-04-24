#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
	int x, y, day_infected, R, state;
	bool symptoms;
} person;
//might need to do this for it to compile,  just in case, it's here

extern void covid_initMaster(unsigned int pop_size, size_t world_width, size_t world_height);
extern bool covid_kernelLaunch(person** d_population, person** d_result, size_t world_width, size_t world_height, size_t pop_size, size_t iterations, int myrank, int numranks);


person *d_population=NULL;
person *d_result=NULL; //


int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    unsigned int pop_size, world_width, world_height, infection_radius, infection_chance, days, ranks;

	

    if( argc != 6 )
    {
	printf("This requires 5 arguments in its current form\n");
	exit(-1);
    }
    
    pop_size = atoi(argv[1]);
	world_width = atoi(argv[2]);
	world_height = atoi(argv[3]);
	infection_radius = atoi(argv[4]);
	days = atoi (argv[5]);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    //Even distribution of pops as of right now
	int popsPerRank;
	popsPerRank = pop_size/num_processes    
    
    /*
	Allocate My Rank’s chunk of the universe*/
    int amountRows = world_height / num_processes; //# of rows of the chunk, each row being world_width long
    covid_initMaster( popsPerRank, world_width, amountRows+2 );


    //alright let's try to output some stuff
    MPI_File filename;


    
	for( int i = 0; i <days*24; i++)
	{
		//printf("started iteration %d of for loop, on processor %d\n", i, world_rank);
	/*Exchange row data with MPI Ranks
	using MPI_Isend/Irecv.*/
		MPI_Request requests0;
		MPI_Request requests1;
		MPI_Request requests2;
		MPI_Request requests3;

		if(world_rank!=0){
			MPI_Irecv( &(d_population[0]),                       world_width, MPI_INT,       (world_rank-1)%num_processes,              1, MPI_COMM_WORLD, &requests0 );
		
		}
		if(world_rank!=num_processes-1){
			MPI_Irecv( &(d_population[world_width*(amountRows-1)]),    		 world_width, MPI_INT,	    (world_rank+1)%num_processes,              1, MPI_COMM_WORLD,&requests1 );
		
		}
		if(world_rank!=0){
			MPI_Isend( &(d_population[world_width]),               world_width, MPI_INT,      (num_processes+world_rank-1)%num_processes, 1, MPI_COMM_WORLD,&requests2 );
		}

		if(world_rank!=num_processes-1){
			MPI_Isend( &(d_population[world_width*(amountRows-2)]),    world_width, MPI_INT,      (world_rank+1)%num_processes,               1, MPI_COMM_WORLD,&requests3 );
		}
		
		MPI_Wait(&requests0, MPI_SUCCESS);
		MPI_Wait(&requests1, MPI_SUCCESS);
		MPI_Wait(&requests2, MPI_SUCCESS);
		MPI_Wait(&requests3, MPI_SUCCESS);
		//printf("Done waiting\n");

		bool ret;
		
		ret = covid_kernelLaunch( &d_population, &d_result, world_width, amount+2, popsPerRank, 1, world_rank, num_processes); 
		
		if(days%24==23){
			//write to file
			MPI_File_open(MPI_COMM_WORLD, "covid.txt",/*???*/,MPI_INFO_NULL, &filename);
			MPI_File_write(filename,/*??*/);
			MPI_File_close(&filename);
		}
		
	}
    

	MPI_Finalize();
	

}
