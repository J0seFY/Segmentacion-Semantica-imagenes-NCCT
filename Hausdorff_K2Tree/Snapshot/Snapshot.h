/*
 * LKTree.h
 *  Es una extensión de los ktree que permite
 *  el etiquetado de aristas.
 *  la idea es poder etiquetar con una o más etiquetas una arista.
 *  cada valor de 32 bits.
 *  Created on: 15-11-2012
 *      Author: Miguel Romero Vásquez
 */

#ifndef LKTREE_H_
#define LKTREE_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <BitSequence.h>
using namespace cds_static;

#include "../Util/LinkedList.h"
#include "../Util/ListIterator.h"
#include "../Util/LinkedList.h"
#include "../Util/HashMap.h"
#include "../Util/Queue.h"
#include "../Util/ObjectTimePosition.h"
#include "../Util/Point.h"
#include "LabelsMap.h"

#ifndef uint
#define uint unsigned int
#endif

#ifndef K
#define K 2
#endif

void detectMemoryCorrupt();

//Representación temporal usada en la construcción del arbol
//pero no es compacta.

typedef struct lnode {
	uint data;
	//lista de etiquetas
	LinkedList ** labels;
	struct lnode** child;
} L_NODE;

typedef struct slkt {
	L_NODE * root;
	int max_Level;
	uint numberNodes;
	uint numberLeaves;
	uint numberTotalLeaves;
	uint totalLabels;
} lkt;
L_NODE * createL_Node();
lkt * createLKTree(uint maxlevels);
void insertNode(lkt * tree, int x, int y, uint label);
void _insertNode(lkt * tree, int x, int y, uint * labelArray);
int destroyLKTree(lkt * tree);


typedef struct matrixRep2
{
    BitSequence * bt;		//Bitmap representando el arbol
    BitSequence * bn; 		//Bitmap representando el padre de las hojas
    BitSequence * bl;     	//Bitmap representando las hojas
    int maxLevel;			//Nivel maximo del arbol
    size_t numberOfNodes;
    size_t numberOfEdges;
    unsigned long long int  * div_level_table;
}MREP2;

//reprecentación compacta de un lkt.
typedef struct sSnapshot {
	uint totObj;
	MREP2 * ktree;
	LabelsMap * labels;
} Snapshot;

Snapshot * createSnapshot(lkt * tree, uint numberOfNodes,
								uint numberOfEdges, uint totObject);


int loadSnapshotFromFile(Snapshot **snapshot, const char *filename);
int loadMREP2FromFile(MREP2 **ktree, FILE *file);
int saveSnapshotToFile(Snapshot *snapshot, const char *filename) ;
int saveMREP2ToFile(MREP2 *ktree, FILE *file);

LinkedList *  rangeQuery(Snapshot * rep, uint p1, uint p2, uint q1, uint q2);
void rangeQuery(Snapshot * snap, uint p1, uint p2, uint q1, uint q2, uint * &Oid, uint * &X, uint * &Y, uint &n);
void rangeQueryHD(Snapshot * snap, uint p1, uint p2, uint q1, uint q2, uint * &Oid, uint * &X, uint * &Y, uint &n);
void rangeQueryHDOriginal(Snapshot * snap, uint p1, uint p2, uint q1, uint q2, uint * &Oid, uint * &X, uint * &Y, uint &n) ;
Point * getObjectPos(Snapshot * rep, uint oid);
void NN(Snapshot * rep, Point  q, uint &oid, Point & p);
void kNN(Snapshot * rep, Point q, uint & kn, uint * oid, Point * p);
void kNNFilter(Snapshot * rep, Point q, uint maxDesp ,uint & kn, uint * oid, Point * p);
void destroySnapshot(Snapshot * rep);
//implementación de la distancia cd hausdorff, tesis Fernando Domínguez
double hausdorffDist(Snapshot * A, Snapshot * B);
double hausdorffDistHDK2(Snapshot * C, Snapshot * D);
double hausdorffDistRandom(Snapshot * E, Snapshot * F);
double hausdorffDistSinCmax(Snapshot * G, Snapshot * H);
void hausdorffDistVersionNueva(Snapshot * G, Snapshot * H);
void hausdorffDistHDK3(Snapshot * G, Snapshot * H);
double hausdorffDistNaive(Snapshot * A, Snapshot * B, int nelementos);
double hausdorffDistTaha(Snapshot * A, Snapshot * B, int nelementos);
/* AGREGADOS POR FERNANDO SANTOLAYA PARA EXPERIMENTOS */
double hausdorffDistHDK3Fila(Snapshot * A, Snapshot * B);
double hausdorffDistHDK3Pila(Snapshot * A, Snapshot * B);
double hausdorffDistHDK3MinHeap(Snapshot * A, Snapshot * B);
double hausdorffDistHDK3MaxHeap(Snapshot * A, Snapshot * B);
double hausdorffDistHDK3MaxHeapv2(Snapshot * A, Snapshot * B);
extern int heapMaxElements;
extern int exitHeapElements;

// Declaración de la versión simétrica utilizada por los bindings de Python
// (definida en Snapshot.cpp)
double SymHDK2tree(Snapshot *A, Snapshot *B);

int getHeapMaxElements();
int getHeapExitElements();

std::vector<Point> extractPointK2tree(Snapshot *J, int nelementos);
Point* extractPointK2tree2(Snapshot *J, int nelementos);
double hausdorffDistTaha2( vector<Point> &A,  vector<Point> &B);
double abdelHDD2( std::vector<Point>& Er,  std::vector<Point>& Br);

size_t sizeSnapshot(Snapshot * lktrep);
size_t sizeMREP(MREP2 * rep);

#endif /* LKTREE_H_ */
