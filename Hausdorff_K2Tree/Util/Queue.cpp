/*
 * Queue.cpp
 *
 *  Created on: 27-11-2012
 *      Author: miguel
 */

#include "Queue.h"

Queue * createEmptyQueue() {
	Queue * q = (Queue *) malloc(sizeof(Queue));
	q->listOfElement = cLinkedList();
	return q;
}
void destroyQueue(Queue * q) {
	destroyLinkedList(q->listOfElement);
	free(q->listOfElement);
	q->listOfElement = NULL;
	free(q);
	q=NULL;
}

int queueSize(Queue * q) {
	return linkedListSize(q->listOfElement);
}

int queueIsEmpty(Queue * q) {
	return empty(q->listOfElement);
}
void enqueue(Queue * q, void * x) {
	addLast(q->listOfElement, cElement(x));
}




void * dequeue(Queue * q) {
	Element * x;
	void * pt;
	x = removeFirst(q->listOfElement);
	pt = x->pt;
	x->pt = NULL;
	free(x);
	x=NULL;
	return pt;
}

