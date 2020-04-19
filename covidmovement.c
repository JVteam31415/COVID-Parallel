//covidmovement.c
#include <stdio.h>
#include <stdlib.h>

int* noRestraints( int* location, int time ){
	//in which a pseudorandom location is chosen as the next 

	int* newloc = (int*)malloc(2*sizeof(int));

	int x = (location[0]*location[1]+1-time)%7;

	if(x==0){
		newloc[0] = location[0]+time%3-1;
		newloc[1] = location[1]-1;
	}
	else if(x==1){
		newloc[0] = location[0]+1;
		newloc[1] = location[1]+1;

	}
	else if(x==2){
		newloc[0] = location[0]+time%3-1;
		newloc[1] = location[1]-1;
		
	}
	else if(x==3){
		newloc[0] = location[0]-1;
		newloc[1] = location[1]+time%3-1;
		
	}
	else if(x==4){
		newloc[0] = location[0]-1;
		newloc[1] = location[1];

	}
	else if(x==6){
		newloc[0] = location[0]+time%3-1;
		newloc[1] = location[1];
	}
	else{
		newloc[0] = location[0]+time%3-1;
		newloc[1] = location[1]-time%3+1;
	}

	return newloc;

}


int* avoid(int*location, int* nearest){
	int* newloc = (int*)malloc(2*sizeof(int));

	int leftright = location[0]-nearest[0];
	int updown = location[1]-nearest[1];

	if(leftright>1){
		newloc[0] = location[0]+1;
	}
	else{
		newloc[0] = location[0]-1;
	}

	if(updown>1){
		newloc[1] = location[1]+1;
	}
	else{
		newloc[1] = location[1]-1;
	}

	return newloc;
}

int* central(int*location, int* cen, int time){
	int* newloc = (int*)malloc(2*sizeof(int));
	int leftright = location[0]-cen[0];
	int updown = location[1]-cen[1];

	if (leftright*leftright+updown*updown <20){
		//chance to leave
		if(leftright+time*updown%6 == 2){
			newloc[0] = location[0] + 30*(time+updown)%5 - 30;
			newloc[1] = location[1] - 30*(time+updown)%5 + 30;
		}
		else{
			newloc = noRestraints(location, time);
		}
	}
	else{
		if(leftright+time*updown%6 == 2){
			newloc[0] = cen[0]+time%3-1;
			newloc[1] = cen[0]-(time+leftright)%3+1;
		}
		else{
			newloc = noRestraints(location, time);
		}

	}
	return newloc;

}

int main(int argc, char** args){

	int* location = (int*)malloc(2*sizeof(int));

	location[0] = 0;
	location[1] =2;

	for( int t=0;t<50;t++){
		int* newloc = noRestraints(location, t);
		location = newloc;
		printf("%d,%d\n", location[0], location[1]);
	}
	


	return 0;
}