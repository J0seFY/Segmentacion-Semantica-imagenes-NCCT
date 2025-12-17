/*
 * TimeMesure.h
 *
 *  Created on: 30-05-2012
 *
 */

#ifndef TIMEMESURE_H_
#define TIMEMESURE_H_

#include <sys/resource.h>
#include <sys/time.h>

typedef struct scronometer {
	double utime0, stime0, wtime0, utime1, stime1, wtime1;
} Cronometer;

Cronometer* cCronometer();
//inicia o reinicia el cronómetro cronom
void start_clock(Cronometer* cronom);

//detiene el cronómetro y devuelve el timepo de CPU (user+system) en segundos
//si el cronometro no se ha inciado retorna un valor negativo.
double stop_clock(Cronometer* cronom);

//entrega los tiempos si el cronometro esta detenido, de otro modo entrega -1
double userTime(Cronometer* cronom);
double sysTime(Cronometer* cronom);
double wallTime(Cronometer* cronom);

#endif /* TIMEMESURE_H_ */
