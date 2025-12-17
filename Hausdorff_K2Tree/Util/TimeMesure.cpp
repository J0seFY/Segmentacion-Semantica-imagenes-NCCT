/*
 * TimeMesure.cpp
 *
 *  Created on: 30-05-2012
 *      Author: miguel
 */
#include "TimeMesure.h"
#include <stdlib.h>

void uswtime(double *usertime, double *systime, double *walltime) {
	//double mega = 1.0e-6;
	struct rusage buffer;
	struct timeval tp;
	struct timezone tzp;

	getrusage(RUSAGE_SELF, &buffer);
	gettimeofday(&tp, &tzp);
	*usertime = (double) buffer.ru_utime.tv_sec
			+ 1.0e-6 * buffer.ru_utime.tv_usec;
	*systime = (double) buffer.ru_stime.tv_sec
			+ 1.0e-6 * buffer.ru_stime.tv_usec;
	*walltime = (double) tp.tv_sec + 1.0e-6 * tp.tv_usec;

}
Cronometer* cCronometer(){
	Cronometer* c=(Cronometer*)malloc(sizeof(Cronometer));
	c->utime0=0;c->utime1=0;
	c->stime0=0;c->stime1=0;
	c->wtime0=0;c->wtime1=0;
	return c;
}

//inicia o reinicia el cronómetro cronom
void start_clock(Cronometer* cronom){
	uswtime(&cronom->utime0,&cronom->stime0, &cronom->wtime0);
}

//detiene el cronómetro y devuelve el timepo de CPU (user+system) en segundos
//si el cronometro no se ha inciado retorna un valor negativo.
double stop_clock(Cronometer* cronom){
	uswtime(&cronom->utime1,&cronom->stime1, &cronom->wtime1);
	return (cronom->utime1 - cronom->utime0 + cronom->stime1 - cronom->stime0);
}

//entrega los tiempos si el cronometro esta detenido, de otro modo entrega -1
double userTime(Cronometer* cronom){;

	return (cronom->utime1!=0)?cronom->utime1-cronom->utime0:-1;
}
double sysTime(Cronometer* cronom){
	return (cronom->stime1!=0)?cronom->stime1-cronom->stime0:-1;
}
double wallTime(Cronometer* cronom){
	return (cronom->wtime1!=0)?cronom->wtime1-cronom->wtime0:-1;
}





