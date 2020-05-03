#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

typedef struct
{
	int x, y, time_infected, R, state;
	bool symptoms;
} person;
//might need to do this for it to compile,  just in case, it's here

extern void covid_initMaster(unsigned int pop_size, size_t world_width, size_t world_height);

extern bool covid_kernelLaunch(
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
);
//extern bool covid_kernelLaunch(person** d_population, person** d_result, size_t world_width, size_t world_height, size_t pop_size, int time, unsigned int radius, float infect_chance, unsigned int recover, int myrank, int numranks);

person *d_population=NULL;
person *d_result=NULL; //


int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    unsigned int pop_size, world_width, world_height, infection_radius,  days,  recovery_time, threshold, behavior1, behavior2;
    float infect_chance, symptom_chance;
	

    if( argc != 12 )
    {
	printf("This requires 11 arguments in its current form\n");
	exit(-1);
    }
    
    pop_size = atoi(argv[1]);
	world_width = atoi(argv[2]);
	world_height = atoi(argv[3]);
	days = atoi (argv[4]);

	infection_radius = atoi(argv[5]);
    infect_chance = atof(argv[6]);
    symptom_chance = atof(argv[7]);
    recovery_time = atoi(argv[8]);

    threshold = atoi(argv[9]);
    behavior1 = atoi(argv[10]);
    behavior2 = atoi(argv[11]); 


    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    //Even distribution of pops as of right now
	int popsPerRank;
	popsPerRank = pop_size/num_processes;
    
    /*
	Allocate My Rank’s chunk of the universe*/
    int amountRows = world_height / num_processes; //# of rows of the chunk, each row being world_width long
    covid_initMaster( popsPerRank, world_width, amountRows+2 );


    //alright let's try to output some stuff
    MPI_File filename;

    MPI_File_open(MPI_COMM_WORLD, "covid.txt", MPI_MODE_CREATE|MPI_MODE_RDWR,  MPI_INFO_NULL, &filename);
    MPI_File_close(&filename);
    
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
		
		ret = covid_kernelLaunch( &d_population, &d_result, world_width, amountRows+2, popsPerRank, i, infection_radius, infect_chance, symptom_chance, recovery_time, threshold, behavior1, behavior2, world_rank, num_processes); 
		
		if(days%24==23){
			//write to file
			MPI_File_open(MPI_COMM_WORLD, "covid.txt",MPI_MODE_RDWR,MPI_INFO_NULL, &filename);
			char* toPrint;
			toPrint = (char*)malloc((amountRows*world_width)*sizeof(char));
			for(int j=0;j<amountRows*world_width;j++){
				/*if(d_population[j] == NULL){
					toPrint[j]=' ';
				}
				else*/ if (d_population[j].state == 0 ){
					toPrint[j] = 'S';
				}
				else if(d_population[j].state == 1 ){
					toPrint[j] = 'I';
				}
				else{
					toPrint[j] = 'R';
				}
			}
			MPI_File_write(filename,toPrint  ,amountRows*world_width, MPI_INT,MPI_STATUS_IGNORE);
			MPI_File_close(&filename);
		}
		
	}
    

	MPI_Finalize();
	

}