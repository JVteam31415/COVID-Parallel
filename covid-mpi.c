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

extern void covid_initMaster(unsigned int pop_size, size_t world_width, size_t world_height, person** d_population);

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
    printf("%d: num_processes: %d\n", world_rank, num_processes);

    //Even distribution of pops as of right now
	int popsPerRank;
	popsPerRank = pop_size/num_processes;
    if (world_rank == 0) {
        popsPerRank += pop_size % num_processes;
    }
    
    printf("%d: popsPerRank: %d\n", world_rank, popsPerRank);
    /*
	Allocate My Rank’s chunk of the universe*/
    int amountRows = world_height / num_processes; //# of rows of the chunk, each row being world_width long
    // account for height not divisible by num_processes
    if (world_rank == 0) {
        amountRows += world_height % num_processes;
    }
    printf("%d: amountRows: %d\n", world_rank, amountRows);


    covid_initMaster( popsPerRank, world_width, amountRows, &d_population );


    int numPopsHere = popsPerRank;

    //alright let's try to output some stuff
    MPI_File filename;

    MPI_File_open(MPI_COMM_WORLD, "covid.txt", MPI_MODE_CREATE|MPI_MODE_RDWR,  MPI_INFO_NULL, &filename);
    MPI_File_close(&filename);
    
    printf("%d: file opened\n", world_rank);


    int lowestRow = (world_rank*(amountRows+1) )-1;
    int highestRow = world_rank*amountRows;
    int highestRowHere = 0;
    int lowestRowHere = amountRows-1;
    printf("%d: high: %d, low: %d\n", world_rank, highestRowHere, lowestRowHere);

    printf("%d: %d\n", world_rank, d_population[0].y);

    for (int k = 0; k < popsPerRank; ++k) {
        printf("%d: p: %d x: %d, y: %d, high: %d, low: %d\n", world_rank, k, d_population[k].x, d_population[k].y, highestRowHere, lowestRowHere);
    }

	for( int i = 0; i <days*24; i++)
	{
		printf("started iteration %d of for loop, on processor %d\n", i, world_rank);
	/*Exchange row data with MPI Ranks
	using MPI_Isend/Irecv.*/
		MPI_Request requests0;
		MPI_Request requests1;
		MPI_Request requests2;
		MPI_Request requests3;

		/*
		1) this node has a list of people
		2) it sends a list of its people that are on the top row of its area 
		and its area's bottom row to the nodes above and below (this does not remove them from its own list)
		3)it then receives a list of the people the row above its area and a list of the people the row below it
		*/
		printf("row 134 %d: finished requests\n", world_rank);
		int top = 0;
		int bot = 0;
		for (int j=0;j<numPopsHere;j++){
			if(d_population[j].y<=highestRowHere){
				top++;
			}
			else if(d_population[j].y>=lowestRowHere){
				bot++;
			}
		}

		person* topRow = (person*)malloc(top*sizeof(person));
		person* botRow = (person*)malloc(bot*sizeof(person));

		printf("row 149 %d: top: %d, bot: %d\n", world_rank, top, bot);
		int tc = 0;
		int bc = 0;
		int counter = 0;
		while( (tc<top)&&(bc<bot)){
			if(d_population[counter].y<=highestRowHere){
				topRow[tc]=d_population[counter];
				tc++;
			}
			else if(d_population[counter].y>=lowestRowHere){
				botRow[bc]=d_population[counter];
				bc++;
			}
			counter++;
		}

		printf("row 165 %d: populated topRow, botRow\n", world_rank);

		/* 
		4)It adds the people in the list it receives to ITS OWN LIST OF PEOPLE FOR THE KERNEL 
		5)Kernel operates, and gives all people in this node's list a new location
		6)remove people outside boundaries from the node's list of people
		*/
		int upperBot;
		int lowerTop;
		if(world_rank!=0){
			MPI_Irecv( &(upperBot), 1, MPI_INT, (world_rank+1)%num_processes, 1, MPI_COMM_WORLD, &requests0 );
		
		}
		printf("%d: 1 finished\n", world_rank);
		if(world_rank!=num_processes-1){
			MPI_Irecv( &(lowerTop), 1, MPI_INT, (world_rank-1)%num_processes, 1, MPI_COMM_WORLD,&requests1 ); //size?
		
		}
		printf("%d: 2 finished\n", world_rank);
		if(world_rank!=0){
			MPI_Isend( &top, 1, MPI_INT, (num_processes+world_rank-1)%num_processes, 1, MPI_COMM_WORLD,&requests2 );
		}

		printf("%d: 3 finished\n", world_rank);
		if(world_rank!=num_processes-1){
			MPI_Isend( &bot, 1, MPI_INT, (world_rank+1)%num_processes, 1, MPI_COMM_WORLD,&requests3 );
		}
		printf("%d: 4 finished\n", world_rank);


		MPI_Wait(&requests0, MPI_SUCCESS);
		printf("%d: countwait0\n", world_rank);
		MPI_Wait(&requests1, MPI_SUCCESS);
		printf("%d: countwait1\n", world_rank);
		MPI_Wait(&requests2, MPI_SUCCESS);
		printf("%d: countwait2\n", world_rank);
		MPI_Wait(&requests3, MPI_SUCCESS);
		printf("%d: countwait3\n", world_rank);

		person* lowerTopRow= (person*)malloc(lowerTop*sizeof(person));
		printf("%d: lowerTopRow allocated\n", world_rank);
		person* upperBotRow = (person*)malloc(upperBot*sizeof(person));

        printf("%d: upperBot: %d\n", world_rank, upperBot);
        printf("%d: lowerTop: %d\n", world_rank, lowerTop);


		if(world_rank!=0 && upperBot > 0){
			MPI_Irecv( (upperBotRow), upperBot, MPI_INT, (world_rank-1)%num_processes, 1, MPI_COMM_WORLD, &requests0 );
		
		}
		printf("%d: first one finished\n", world_rank);
		if(world_rank!=num_processes-1 && lowerTop > 0){
			MPI_Irecv( (lowerTopRow), lowerTop, MPI_INT, (world_rank+1)%num_processes, 1, MPI_COMM_WORLD,&requests1 ); //size?
		
		}
		printf("%d: second one finished\n", world_rank);
		if(world_rank!=0 && top > 0){
			MPI_Isend( (topRow), top, MPI_INT, (num_processes+world_rank-1)%num_processes, 1, MPI_COMM_WORLD,&requests2 );
		}

		printf("%d: third one finished\n", world_rank);
		if(world_rank!=num_processes-1 && bot > 0){
			MPI_Isend( (botRow), bot, MPI_INT, (world_rank+1)%num_processes, 1, MPI_COMM_WORLD,&requests3 );
		}

		printf("%d: fourth one finished\n", world_rank);
		
		MPI_Wait(&requests0, MPI_SUCCESS);
        printf("%d: wait1\n", world_rank);
		MPI_Wait(&requests1, MPI_SUCCESS);
        printf("%d: wait2\n", world_rank);
		MPI_Wait(&requests2, MPI_SUCCESS);
        printf("%d: wait3\n", world_rank);
		MPI_Wait(&requests3, MPI_SUCCESS);
		printf("%d: Done waiting\n", world_rank);

		bool ret;

		person* newPopulation = (person*)malloc( (numPopsHere+upperBot+lowerTop)*sizeof(person));
		for(int j=0;j<upperBot;j++){
			upperBotRow[j].y = -1;
			newPopulation[j] = upperBotRow[j];
		} 

		for(int j=0;j<lowerTop;j++){
			lowerTopRow[j].y = amountRows;
			newPopulation[upperBot+j] = lowerTopRow[j];
		} 

		for(int j=0;j<numPopsHere;j++){
			newPopulation[upperBot+lowerTop+j] = d_population[j];
		} 

        printf("%d: new population done\n", world_rank);

		d_population = newPopulation;
        printf("%d: before kernel\n", world_rank);		
		ret = covid_kernelLaunch( &d_population, &d_result, world_width, amountRows+2, popsPerRank, i, infection_radius, infect_chance, symptom_chance, recovery_time, threshold, behavior1, behavior2, world_rank, num_processes); 
        printf("%d: after kernel\n", world_rank);		
		
        // remove people that moved outside
        person *actualPopulation;
		int actual = 0;

		for (int j=0; j<upperBot+lowerTop+numPopsHere;j++ ){

			if(d_population[j].y < highestRowHere){

			}
			else if(d_population[j].y > lowestRowHere){

			}
			else{
				actual++;
			}
		}
        printf("%d: counted actual pop\n", world_rank);		
		actualPopulation = (person*)malloc(sizeof(person)*actual);
		actual = 0;
		for (int j=0; j<upperBot+lowerTop+numPopsHere;j++ ){

			if(d_population[j].y < highestRowHere){

			}
			else if(d_population[j].y > lowestRowHere){

			}
			else{
				actualPopulation[actual] = d_population[j];
				actual++;
			}
		}
		numPopsHere = actual;
		d_population = actualPopulation;
        printf("%d: inserted actual pop\n", world_rank);		


		if(days%24==23){
			//write to file
			MPI_File_open(MPI_COMM_WORLD, "covid.txt",MPI_MODE_RDWR,MPI_INFO_NULL, &filename);
			char* toPrint;
			toPrint = (char*)malloc((amountRows*world_width)*sizeof(char));
			for(int j=0;j<numPopsHere;j++){
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
			printf("%d: Printed out %d day\n", world_rank, i/24);
		}
		
	}
    

	MPI_Finalize();
	

}