#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>

typedef struct
{
	int x, y, time_infected, R, state, symptoms;
    // bool symptoms;
} person;
//might need to do this for it to compile,  just in case, it's here

extern void covid_initMaster(unsigned int pop_size, size_t world_width, size_t world_height, person** d_population, person** d_result, int myrank, int max_pop);

extern void setup_kernelLaunch();

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

int numPlaces (int n) {
    if (n < 0) n = (n == INT_MIN) ? INT_MAX : -n;
    if (n < 10) return 1;
    if (n < 100) return 2;
    if (n < 1000) return 3;
    if (n < 10000) return 4;
    if (n < 100000) return 5;
    if (n < 1000000) return 6;
    if (n < 10000000) return 7;
    if (n < 100000000) return 8;
    if (n < 1000000000) return 9;
    /*      2147483647 is 2^31-1 - add more ifs as needed
 *             and adjust this final return as well. */
    return 10;
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    double t1 = MPI_Wtime();

    unsigned int pop_size, world_width, world_height, infection_radius,  days,  recovery_time, threshold, behavior1, behavior2;
    float infect_chance, symptom_chance;

    if( argc != 12 )
    {
//	printf("This requires 11 arguments in its current form\n");
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

    // make MPI type for person
    const int nitems = 6;
    int blocklengths[6] = {1,1,1,1,1,1};
    // MPI_Datatype types[6] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_C_BOOL};
    MPI_Datatype types[6] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Datatype mpi_person_type;
    MPI_Aint offsets[6];

    offsets[0] = offsetof(person, x);
    offsets[1] = offsetof(person, y);
    offsets[2] = offsetof(person, time_infected);
    offsets[3] = offsetof(person, R);
    offsets[4] = offsetof(person, state);
    offsets[5] = offsetof(person, symptoms);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_person_type);
    MPI_Type_commit(&mpi_person_type);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
//    printf("%d: num_processes: %d\n", world_rank, num_processes);

    //Even distribution of pops as of right now
	int popsPerRank;
	popsPerRank = pop_size/num_processes;
    if (world_rank == 0) {
        popsPerRank += pop_size % num_processes;
    }
    
//    printf("%d: popsPerRank: %d\n", world_rank, popsPerRank);
    /*
	Allocate My Rank’s chunk of the universe*/
    int amountRows = world_height / num_processes; //# of rows of the chunk, each row being world_width long
    // account for height not divisible by num_processes
    if (world_rank == 0) {
        amountRows += world_height % num_processes;
    }
//    printf("%d: amountRows: %d\n", world_rank, amountRows);

//    printf("%d: before initMaster\n", world_rank);
//==================================================================================================================
    covid_initMaster( popsPerRank, world_width, amountRows, &d_population, &d_result, world_rank, pop_size);
//==================================================================================================================
//    printf("%d: after initMaster, before setup_kernelLaunch\n", world_rank);
    setup_kernelLaunch();
//    printf("%d: after setup_kernelLaunch\n", world_rank);

    int numPopsHere = popsPerRank;

    //alright let's try to output some stuff
    MPI_File filename;

    MPI_File_open(MPI_COMM_WORLD, "covid.txt", MPI_MODE_CREATE|MPI_MODE_RDWR,  MPI_INFO_NULL, &filename);
    MPI_File_close(&filename);
    
//    printf("%d: file opened\n", world_rank);


    int lowestRow = (world_rank*(amountRows+1) )-1;
    int highestRow = world_rank*amountRows;
    int highestRowHere = 0;
    int lowestRowHere = amountRows-1;
//    printf("%d: high: %d, low: %d\n", world_rank, highestRowHere, lowestRowHere);

//    printf("%d: %d\n", world_rank, d_population[0].y);

    for (int k = 0; k < popsPerRank; ++k) {
//        printf("%d: p: %d x: %d, y: %d, high: %d, low: %d\n", world_rank, k, d_population[k].x, d_population[k].y, highestRowHere, lowestRowHere);
    }

	for( int i = 0; i <days; i++)
	{
//		printf("started iteration %d of for loop, on processor %d\n", i, world_rank);
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
//		printf("row 134 %d: finished requests\n", world_rank);
		int top = 0;
		int bot = 0;
		for (int j=0;j<numPopsHere;j++){
			if(d_population[j].y<=highestRowHere){
  //              printf("%d: TOP COUNT y: %d <= %d\n", world_rank, d_population[j].y, highestRowHere);
				top++;
			}
			else if(d_population[j].y>=lowestRowHere){
//                printf("%d: BOT COUNT y: %d >= %d\n", world_rank, d_population[j].y, lowestRowHere);
				bot++;
			}
		}

		person* topRow = (person*)malloc(top*sizeof(person));
		person* botRow = (person*)malloc(bot*sizeof(person));

//		printf("row 149 %d: top: %d, bot: %d\n", world_rank, top, bot);
		int tc = 0;
		int bc = 0;
		int counter = 0;
		while( (tc<top)||(bc<bot)){
            //printf("%d: counter: %d, numPopsHere: %d\n", world_rank, counter, numPopsHere);
			if(d_population[counter].y<=highestRowHere){
                
//                printf("%d: TOP BUILD counter: %d x: %d, y: %d, state: %d\n", world_rank, counter, d_population[counter].x, d_population[counter].y, d_population[counter].state);

				topRow[tc]=d_population[counter];
				tc++;
			}
			else if(d_population[counter].y>=lowestRowHere){

  //              printf("%d: BOT BUILD counter: %d x: %d, y: %d, state: %d\n", world_rank, counter, d_population[counter].x, d_population[counter].y, d_population[counter].state);

				botRow[bc]=d_population[counter];
				bc++;
			}
			counter++;
		}

        for (int i=0; i < top; ++i) {
//            printf("%d: TOP x: %d, y: %d, state: %d\n", world_rank, topRow[i].x, topRow[i].y, topRow[i].state);
        }
        for (int i=0; i < bot; ++i) {
  //          printf("%d: BOT x: %d, y: %d, state: %d\n", world_rank, botRow[i].x, botRow[i].y, botRow[i].state);
        }


//		printf("row 165 %d: populated topRow, botRow\n", world_rank);

		/* 
		4)It adds the people in the list it receives to ITS OWN LIST OF PEOPLE FOR THE KERNEL 
		5)Kernel operates, and gives all people in this node's list a new location
		6)remove people outside boundaries from the node's list of people
		*/
		int upperBot = 0 ;
		int lowerTop = 0;
		if(world_rank!=0){
			MPI_Irecv( &(upperBot), 1, MPI_INT, (world_rank-1)%num_processes, 1, MPI_COMM_WORLD, &requests0 );
		
		}
//		printf("%d: 1 finished\n", world_rank);
		if(world_rank!=num_processes-1){
			MPI_Irecv( &(lowerTop), 1, MPI_INT, (world_rank+1)%num_processes, 1, MPI_COMM_WORLD,&requests1 ); //size?
		
		}
//		printf("%d: 2 finished\n", world_rank);
		if(world_rank!=0){
			MPI_Isend( &top, 1, MPI_INT, (num_processes+world_rank-1)%num_processes, 1, MPI_COMM_WORLD,&requests2 );
		}

//		printf("%d: 3 finished\n", world_rank);
		if(world_rank!=num_processes-1){
			MPI_Isend( &bot, 1, MPI_INT, (world_rank+1)%num_processes, 1, MPI_COMM_WORLD,&requests3 );
		}
//		printf("%d: 4 finished\n", world_rank);


		if(world_rank!=0){
			MPI_Wait(&requests0, MPI_SUCCESS);
//			printf("%d: countwait0\n", world_rank);
			MPI_Wait(&requests2, MPI_SUCCESS);
//			printf("%d: countwait2\n", world_rank);
		}

		if(world_rank!=num_processes-1){
			MPI_Wait(&requests1, MPI_SUCCESS);
//			printf("%d: countwait1\n", world_rank);
			MPI_Wait(&requests3, MPI_SUCCESS);
//			printf("%d: countwait3\n", world_rank);
		}

		person* lowerTopRow= (person*)malloc(lowerTop*sizeof(person));
//		printf("%d: lowerTopRow allocated\n", world_rank);
		person* upperBotRow = (person*)malloc(upperBot*sizeof(person));

//        printf("%d: upperBot: %d\n", world_rank, upperBot);
//        printf("%d: lowerTop: %d\n", world_rank, lowerTop);



		MPI_Request row_requests0;
		MPI_Request row_requests1;
		MPI_Request row_requests2;
		MPI_Request row_requests3;

		if(world_rank!=0 && upperBot > 0){
			MPI_Irecv( (upperBotRow), upperBot, mpi_person_type, (world_rank-1)%num_processes, 1, MPI_COMM_WORLD, &row_requests0 );
		
		}
//		printf("%d: first one finished\n", world_rank);
		if(world_rank!=num_processes-1 && lowerTop > 0){
			MPI_Irecv( (lowerTopRow), lowerTop, mpi_person_type, (world_rank+1)%num_processes, 1, MPI_COMM_WORLD,&row_requests1 ); //size?
		
		}
//		printf("%d: second one finished\n", world_rank);
		if(world_rank!=0 && top > 0){
			MPI_Isend( (topRow), top, mpi_person_type, (num_processes+world_rank-1)%num_processes, 1, MPI_COMM_WORLD,&row_requests2 );
		}

//		printf("%d: third one finished\n", world_rank);
		if(world_rank!=num_processes-1 && bot > 0){
			MPI_Isend( (botRow), bot, mpi_person_type, (world_rank+1)%num_processes, 1, MPI_COMM_WORLD,&row_requests3 );
		}

//		printf("%d: fourth one finished\n", world_rank);
		
		if(world_rank!=0 && upperBot > 0){
			MPI_Wait(&row_requests0, MPI_SUCCESS);
//        	printf("%d: wait1\n", world_rank);
    	}
    	if(world_rank!=num_processes-1 && lowerTop > 0){
			MPI_Wait(&row_requests1, MPI_SUCCESS);
//        	printf("%d: wait2\n", world_rank);
        }
        if(world_rank!=0 && top > 0){
			MPI_Wait(&row_requests2, MPI_SUCCESS);
//	        printf("%d: wait3\n", world_rank);
	    }
	    if(world_rank!=num_processes-1 && bot > 0){
			MPI_Wait(&row_requests3, MPI_SUCCESS);
//			printf("%d: Done waiting\n", world_rank);
		}


        for (int i=0; i < lowerTop; ++i) {
//            printf("%d: LOWERTOP x: %d, y: %d, state: %d\n", world_rank, lowerTopRow[i].x, lowerTopRow[i].y, lowerTopRow[i].state);
        }
        for (int i=0; i < upperBot; ++i) {
//            printf("%d: UPPERBOT x: %d, y: %d, state: %d\n", world_rank, upperBotRow[i].x, upperBotRow[i].y, upperBotRow[i].state);
        }


		bool ret;
        int new_pop_size = numPopsHere + upperBot + lowerTop;
		person* newPopulation = (person*)malloc( new_pop_size*sizeof(person));
        person* newResult = (person*)malloc( new_pop_size*sizeof(person));
		for(int j=0;j<upperBot;j++){
            int h;
            if (world_rank == 1) { //rank above is 0 which has a diff height
                h = amountRows + world_height % num_processes;
		    } else {
                h = amountRows;
            }
            upperBotRow[j].y -= h;
            
			newPopulation[j] = upperBotRow[j];
            newResult[j] = upperBotRow[j];
		} 
//        printf("%d: upperBot added to newPopulation\n", world_rank);
        
		for(int j=0;j<lowerTop;j++){
            int h;
            if (world_rank == 1) { //rank above is 0 which has a diff height
                h = amountRows + world_height % num_processes;
		    } else {
                h = amountRows;
            }
			lowerTopRow[j].y += h;
			newPopulation[upperBot+j] = lowerTopRow[j];
            newResult[upperBot+j] = lowerTopRow[j];
		} 
//        printf("%d: lowerTop added to newPopulation\n", world_rank);

		for(int j=0;j<numPopsHere;j++){
			newPopulation[upperBot+lowerTop+j] = d_population[j];
			newResult[upperBot+lowerTop+j] = d_result[j];
		} 

//        printf("%d: new population done\n", world_rank);

		d_population = newPopulation;
        d_result = newResult;

        person *pb = d_population;
        person *rb = d_result;
        for (int i = 0; i < new_pop_size; ++i) {
//            printf("%d:BEFORE KERNEL pop: x: %d, y: %d, state: %d \t|\t res: x: %d, y: %d, state: %d\n", world_rank, pb[i].x, pb[i].y, pb[i].state, rb[i].x, rb[i].y, rb[i].state);
        }

//        printf("%d: before kernel\n", world_rank);		
		ret = covid_kernelLaunch( &d_population, &d_result, world_width, amountRows+2, new_pop_size, i, infection_radius, infect_chance, symptom_chance, recovery_time, threshold, behavior1, behavior2, world_rank, num_processes); 
//        printf("%d: after kernel\n", world_rank);		
	
        person *p = d_population;
        person *r = d_result;
        for (int i = 0; i < new_pop_size; ++i) {
//            printf("%d:AFTER KERNEL pop: x: %d, y: %d, state: %d \t|\t res: x: %d, y: %d, state: %d\n", world_rank, p[i].x, p[i].y, p[i].state, r[i].x, r[i].y, r[i].state);
        }

        // remove people that moved outside
        person *actualPopulation;
        //person *actualResult;
		int actual = 0;

		for (int j=0; j<upperBot+lowerTop+numPopsHere;j++ ){
            if (d_population[j].y >= highestRowHere && d_population[j].y <= lowestRowHere) {
                actual++;
            }
		}
//        printf("%d: counted actual pop: %d\n", world_rank, actual);		
		actualPopulation = (person*)malloc(sizeof(person)*actual);
		//actual = 0;
		int actual_index = 0;
		for (int j=0; j<upperBot+lowerTop+numPopsHere;j++ ){
            if (d_population[j].y >= highestRowHere && d_population[j].y <= lowestRowHere) {
				actualPopulation[actual_index] = d_population[j];
				actual_index++;
            }
		}
		numPopsHere = actual;
        for (int i=0; i < actual; ++i) {
//            printf("%d: ACTUALPOP x: %d, y: %d, state: %d\n", world_rank, actualPopulation[i].x, actualPopulation[i].y, actualPopulation[i].state);
        }
		d_population = actualPopulation;
        //printf("%d: inserted actual pop\n", world_rank);		


		
		//printf("Printing out\n");
		MPI_File_open(MPI_COMM_WORLD, "covid.txt",MPI_MODE_RDWR,MPI_INFO_NULL, &filename);
		char* toPrint;
        int print_n = 50;
		toPrint = (char*)malloc((numPopsHere*print_n)*sizeof(char));
        int offset = (num_processes* i *pop_size + pop_size * world_rank)* print_n * sizeof(char);
		for(int j=0;j<numPopsHere;j++){
            int p_i = 0;
            char p_block[50];
            sprintf(p_block, "%d,%d,%c,%d,%d\n", world_rank, i, d_population[j].state == 0 ? 'S' : d_population[j].state == 1 ? 'I' : 'R', d_population[j].x, d_population[j].y);
            strcat(toPrint, p_block);
/*
			toPrint[print_n*j+(p_i++)]='0'+world_rank;
            toPrint[print_n*j+(p_i++)]=',';
            toPrint[print_n*j+(p_i++)] = '0'+i;
            toPrint[print_n*j+(p_i++)]=',';
			if (d_population[j].state == 0 ){
				toPrint[print_n*j+(p_i++)] = 'S';
			}
			else if(d_population[j].state == 1 ){
				toPrint[print_n*j+(p_i++)] = 'I';
			}
			else{
				toPrint[print_n*j+(p_i++)] = 'R';
			}
            toPrint[print_n*j+(p_i++)] = ',';
            int x_count = numPlaces(d_population[j].x);
            char x_str[10];
            sprintf(x_str, "%d", d_population[j].x);
            printf("X_STR: %s, XCOUNT: %d\n", x_str, x_count); 
            for (int k=0; k<x_count; ++k){
                toPrint[print_n*j+(p_i++)] = x_str[k];
            }
            toPrint[print_n*j+(p_i++)] = ',';
            int y_count = numPlaces(d_population[j].y);
            char y_str[10]; 
            sprintf(y_str, "%d", d_population[j].y);
            for (int k=60; k<y_count; ++k){
                toPrint[print_n*j+(p_i++)] = y_str[k];
            }
			toPrint[print_n*j+(p_i++)]='\n';
*/
		}
        //toPrint[numPopsHere*print_n] = '\0';

		//printf("%s", toPrint);
		MPI_File_write_at(filename, offset  ,toPrint, strlen(toPrint), MPI_CHAR,MPI_STATUS_IGNORE);
		MPI_File_close(&filename);
		//printf("%d: Printed out %d day\n", world_rank, i/24);
	}
    
    double t2 = MPI_Wtime();

    printf("%f\n", t2-t1);
    
    //MPI_Type_Free(&mpi_person_type);
	MPI_Finalize();
	

}
