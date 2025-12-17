/*
 * Queue.h
 *
 *  Created on: 27-11-2012
 *      Author: miguel
 */

#ifndef QUEUE_H_
#define QUEUE_H_

#include "LinkedList.h"
//operaciones de enqueue y dequeue

typedef struct sQueue {
	LinkedList * listOfElement;
} Queue;

Queue * createEmptyQueue();
void destroyQueue(Queue * q);
int queueSize(Queue * q);
int queueIsEmpty(Queue * q);
void enqueue(Queue * q, void * x);
void * dequeue(Queue * q);

#endif /* QUEUE_H_ */
