/*
 * LKTree.cpp
 *
 *  Created on: 15-11-2012
 *      Author: miguel
 */

#include "Snapshot.h"

#include <libcdsBasics.h>
#include <sys/time.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <limits>
#include <queue>
#include <stack>
#include <vector>

#include "../Util/Element.h"
#include "../Util/Factory.h"

using namespace std;

#define MAX_RESULT_LIMIT 200000
//void detectMemoryCorrupt() {
//	int * i = (int*) malloc(sizeof(int));
//	free(i);
//	int * j = (int*) malloc(sizeof(int));
//	free(j);
//	int * k = (int*) malloc(sizeof(int));
//	free(k);
//	int * l = (int*) malloc(sizeof(int));
//	free(l);
//}
int heapMaxElements = 0;  // Definición
int exitHeapElements = 0;  // Definición


//Operacioens con representación de árbol
//creación de la representación
L_NODE *createL_Node() {
    L_NODE *resp = (L_NODE *) malloc(sizeof(L_NODE));
    resp->data = 0;
    resp->child = NULL;
    resp->labels = NULL;
    return resp;
}

lkt *createLKTree(uint maxlevels) {
    lkt *tree = (lkt *) malloc(sizeof(lkt));
    tree->root = createL_Node();
    tree->max_Level = maxlevels;
    tree->numberNodes = 0;
    tree->numberLeaves = 0;
    tree->numberTotalLeaves = 0;
    tree->totalLabels = 0;
    return tree;
}

//delete a sub-tree recursively
int destroyLNode(L_NODE *node) {
    int resp = 0;
    if (node == NULL) {
        return resp;
    } else {
        if (node->child != NULL) {
            //delete child recursively
            for (int i = 0; i < K * K; i++) {
                resp += destroyLNode(node->child[i]);
                node->child[i] = NULL;
            }
            free(node->child);
            node->child = NULL;
            resp += sizeof(L_NODE *) * K * K + sizeof(L_NODE *) * K * K;
        }
        if (node->labels != NULL) {
            for (int i = 0; i < K * K; i++) {
                if (node->labels[i] != NULL) {
                    resp += sizeof(LinkedList *)
                            + node->labels[i]->size
                              * (sizeof(Node) + sizeof(Element)
                                 + sizeof(uint));
                    destroyLinkedList(node->labels[i]);
                    free(node->labels[i]);
                    node->labels[i] = NULL;

                }
            }
            free(node->labels);
            node->labels = NULL;
        }
        free(node);
        node = NULL;
        resp += sizeof(L_NODE) + sizeof(L_NODE *);
    }
    return resp;
}

//elimina la reprecentación temporal de un lktree.
//retorna el total de bytes liberados
int destroyLKTree(lkt *tree) {
    int resp = 0;
    if (tree == NULL)
        return resp;
    //Elimino la raiz
    resp = destroyLNode(tree->root);
    tree->root = NULL;
    //elimino el nodo cabecera.
    free(tree);
    tree = NULL;
    return resp;
}

void insertNode(lkt *tree, int x, int y, uint label) {
    uint *pt = (uint *) malloc(sizeof(uint));
    *pt = label;
    _insertNode(tree, x, y, pt);

}

// guardar aquí información como rangoX[max..min], rangoY[max..min],
//rangoLabel[max..min] y usar esta información para ajustar los limites
//del k2tree, en vez de usar parámetros. asi es más simple la estructura
//con menos parametros.
//insertaNodo con la etiqueta respectiva

void _insertNode(lkt *tree, int x, int y, uint *labelArray) {
    uint i, node = 0;
    unsigned long long int div_level;
    int l = 0;
    L_NODE *n = tree->root;
    while (l <= tree->max_Level) {

        div_level = pow(K, tree->max_Level - l);
        node = (x / div_level) * K + y / div_level;
        if (l == tree->max_Level) {
            if (n->data == 0) {
                tree->numberLeaves++;
            }
            n->data = n->data | (0x1 << node);   //1;
        } else {
            if (n->child == NULL) {
                if (l < tree->max_Level - 1) {
                    tree->numberNodes += K * K;
                } else {
                    tree->numberTotalLeaves += K * K;
                }
                n->child = (L_NODE **) calloc(K * K, sizeof(L_NODE *));
                for (i = 0; i < K * K; i++) {
                    n->child[i] = NULL;
                }
            }
            if (n->child[node] == NULL) {
                n->child[node] = createL_Node();
            }

            //n apunta al nodo insertado (intermedio)
            n = n->child[node];
        }
        x = x % div_level;
        y = y % div_level;
        l++;
    }
    //aquí n es el último nodo insertado, por lo tanto la hoja.
    //hay que insertar la etiqueta en la lista asosiada al nodo
    //
    tree->totalLabels++;

    if (n->labels == NULL) {
        n->labels = (LinkedList **) calloc(K * K, sizeof(LinkedList *));
    }
    if (n->labels[node] == NULL) {
        //se crea la lista de etiquetas si esta no existe
        n->labels[node] = cLinkedList();
    }
    //aquí es importante que el key de element sea node, este
    //valor será usado en la construcción de la representación
    //compacta.
    add(n->labels[node], cElementKey(node, labelArray));

}

//Operaciones con representación compacta

Snapshot *createSnapshot(lkt *tree, uint numberOfNodes, uint numberOfEdges,
                         uint totObject) {
    //NO destruye lkt, hay que hacerlo desde afuera.
    //Estrategia general:
    //usando el arbol se construye el ktree de los puntos, y se construye un
    //arbol para las etiquetas.  Luego se construye el ktree para las estiquetas
    //con dicho arbol.

    //en este caso los valores de Numero de objetos y numero de arcos no se
    //si son necesarios, hay que revisar.

    Snapshot *Lrep;
    Lrep = (Snapshot *) malloc(sizeof(Snapshot));
    Lrep->totObj = totObject;
    Lrep->ktree = (MREP2 *) malloc(sizeof(MREP2));
    Lrep->ktree->maxLevel = tree->max_Level;
    Lrep->ktree->numberOfNodes = numberOfNodes;
    Lrep->ktree->numberOfEdges = numberOfEdges;
    size_t bits_BT_len = tree->numberNodes;
    size_t bits_BN_len = tree->numberTotalLeaves;
    size_t bits_LI_len = tree->numberLeaves * K * K;

    LinkedList *labelLists = cLinkedList();

    uint *bits_BT = new uint[uint_len(bits_BT_len, 1)]();
    uint *bits_BN = new uint[uint_len(bits_BN_len, 1)]();
    uint *bits_LI = new uint[uint_len(bits_LI_len, 1)]();

    uint k, j, queuecont, conttmp, node, pos = 0;

    unsigned long long div_level;
    int i;
    char isroot = 1;
    Queue *q = createEmptyQueue();
    L_NODE *subTree;
    enqueue(q, tree->root);
    queuecont = 1;

    for (i = 0; i < tree->max_Level; i++) {
        //i:iteración por nivel hasta el penúltimo
        conttmp = 0;
        div_level = pow(K, tree->max_Level - i);
        for (k = 0; k < queuecont; k++) {
            //por elemento de la cola de un mismo nivel
            subTree = (L_NODE *) dequeue(q);
            if ((subTree != NULL) && (subTree->child != NULL)) {
                for (j = 0; j < K * K; j++) {
                    node = j;
                    conttmp++;
                    enqueue(q, subTree->child[node]);
                }
                if (!isroot)
                    bitset(bits_BT, pos);
            }
            if (!isroot)
                pos++;
            isroot = 0;
        }
        queuecont = conttmp;
    }

    Lrep->ktree->bt = PlainBitSequenceFactory(bits_BT, bits_BT_len);
    pos = 0;
    size_t pos_inf = 0;

    //	fprintf(stderr,"Empezando bitmap de hojas utiles\n");
    while (!queueIsEmpty(q)) {
        subTree = (L_NODE *) dequeue(q);
        if ((subTree != NULL) && (subTree->data)) {
            //si no es 0
            bitset(bits_BN, pos);
            //hay que separar las listas

            for (i = 0; i < K * K; i++) {
                if ((subTree->data) & (0x1 << i)) {
                    bitset(bits_LI, pos_inf);
                    add(labelLists, cElement(subTree->labels[i]));
                    labelLists->count += subTree->labels[i]->size;
                }
                pos_inf++;
            }
        }
        pos++;
    }
    destroyQueue(q);
    Lrep->ktree->bn = PlainBitSequenceFactory(bits_BN, bits_BN_len);
    Lrep->ktree->bl = PlainBitSequenceFactory(bits_LI, bits_LI_len);

    Lrep->ktree->div_level_table = (long long uint *) malloc(
            sizeof(long long uint) * (Lrep->ktree->maxLevel + 1));
    for (i = 0; i <= Lrep->ktree->maxLevel; i++)
        Lrep->ktree->div_level_table[i] = pow(K, Lrep->ktree->maxLevel - i);

    //compactando las listas de id.
    Lrep->labels = new LabelsMap(labelLists, totObject);
    //Eliminando los nodos de las listas, pero sin eliminar las listas de ID,
    //para evitar un doble free al eliminar el arbol temporal que también
    //apunta a las mismas listas de objetos.
    destroyLinkedListWhithoutElementPointer(labelLists);
    free(labelLists);
    labelLists = NULL;

    delete[] bits_BT;
    bits_BT = NULL;
    delete[] bits_BN;
    bits_BN = NULL;
    delete[] bits_LI;
    bits_LI = NULL;
    return Lrep;

}

void destroySnapshot(Snapshot *rep) {
    //destroyRepresentation(rep->ktree);
    delete rep->ktree->bl;
    delete rep->ktree->bn;
    delete rep->ktree->bt;
    free(rep->ktree->div_level_table);
    rep->ktree->bl = NULL;
    rep->ktree->bn = NULL;
    rep->ktree->bt = NULL;
    rep->ktree->div_level_table = NULL;
    free(rep->ktree);
    rep->ktree = NULL;

    delete rep->labels;
    rep->labels = NULL;
    free(rep);
    rep = NULL;
}

void recursiveGetRank(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
                      uint dp, uint dq, int x, int l, uint *&Oid, uint *&X,
                      uint *&Y, uint &n) {
    MREP2 *rep = snap->ktree;
    uint i = 0, j, leaf;
    uint y, p1new, p2new, q1new, q2new;
    unsigned long int divlevel;
    if (l == rep->maxLevel) {
        //recorrido por el bitarray leavesInf
        leaf = x + i * p1;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                leaf = x + j;
                if (rep->bl->access(leaf)) {
                    //dp + i => posición X del objeto econtrado
                    //dq + j => posición Y del objeto encontrado.
                    snap->labels->getObjects(rep->bl->rank1(leaf), dp + i,
                                             dq + j, Oid, X, Y, n);
                }
            }
            leaf += K;
        }

    }

    if ((l == rep->maxLevel - 1)
        && (rep->bn->access(x - rep->bt->getLength()))) {
        //recorrido por el bitarray bn
        y = (rep->bn->rank1(x - rep->bt->getLength()) - 1) * K * K;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                recursiveGetRank(snap, 0, 0, 0, 0, dp + i, dq + j,
                                 y + K * i + j, l + 1, Oid, X, Y, n);
            }
        }

    }
    if ((x == -1) || ((l < rep->maxLevel - 1) && (rep->bt->access(x)))) {
        //recorrido por el bitarray bt
        y = (x == -1) ? 0 : rep->bt->rank1(x) * K * K;
        divlevel = rep->div_level_table[l + 1];
        for (i = p1 / divlevel; i <= p2 / divlevel; i++) {
            p1new = 0;
            if (i == p1 / divlevel)
                p1new = p1 % divlevel;
            p2new = divlevel - 1;
            if (i == p2 / divlevel)
                p2new = p2 % divlevel;
            for (j = q1 / divlevel; j <= q2 / divlevel; j++) {
                q1new = 0;
                if (j == q1 / divlevel)
                    q1new = q1 % divlevel;
                q2new = divlevel - 1;
                if (j == q2 / divlevel)
                    q2new = q2 % divlevel;
                recursiveGetRank(snap, p1new, p2new, q1new, q2new,
                                 dp + divlevel * i, dq + divlevel * j, y + K * i + j,
                                 l + 1, Oid, X, Y, n);
            }
        }

    }
}
void recursiveGetRank2(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
                      uint dp, uint dq, int x, int l, uint *&Oid, uint *&X,
                      uint *&Y, uint &n) {
    MREP2 *rep = snap->ktree;
    uint i = 0, j, leaf;
    uint y, p1new, p2new, q1new, q2new;
    unsigned long int divlevel;
    if (l == rep->maxLevel) {
        //recorrido por el bitarray leavesInf
        leaf = x + i * p1;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                leaf = x + j;
                if (rep->bl->access(leaf)) {
                    //dp + i => posición X del objeto econtrado
                    //dq + j => posición Y del objeto encontrado.
                   // snap->labels->getObjects(rep->bl->rank1(leaf), dp + i,
                     //                        dq + j, Oid, X, Y, n);
                    X[n] = dp + i;
                    Y[n] = dq + j;
                    n++;
                }
            }
            leaf += K;
        }

    }

    if ((l == rep->maxLevel - 1)
        && (rep->bn->access(x - rep->bt->getLength()))) {
        //recorrido por el bitarray bn
        y = (rep->bn->rank1(x - rep->bt->getLength()) - 1) * K * K;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                recursiveGetRank2(snap, 0, 0, 0, 0, dp + i, dq + j,
                                 y + K * i + j, l + 1, Oid, X, Y, n);
            }
        }

    }
    if ((x == -1) || ((l < rep->maxLevel - 1) && (rep->bt->access(x)))) {
        //recorrido por el bitarray bt
        y = (x == -1) ? 0 : rep->bt->rank1(x) * K * K;
        divlevel = rep->div_level_table[l + 1];
        for (i = p1 / divlevel; i <= p2 / divlevel; i++) {
            p1new = 0;
            if (i == p1 / divlevel)
                p1new = p1 % divlevel;
            p2new = divlevel - 1;
            if (i == p2 / divlevel)
                p2new = p2 % divlevel;
            for (j = q1 / divlevel; j <= q2 / divlevel; j++) {
                q1new = 0;
                if (j == q1 / divlevel)
                    q1new = q1 % divlevel;
                q2new = divlevel - 1;
                if (j == q2 / divlevel)
                    q2new = q2 % divlevel;
                recursiveGetRank2(snap, p1new, p2new, q1new, q2new,
                                 dp + divlevel * i, dq + divlevel * j, y + K * i + j,
                                 l + 1, Oid, X, Y, n);
            }
        }

    }
}

int saveMREP2ToFile(MREP2 *ktree, std::ofstream &ofs) {
    if (!ktree) return -1;

    // Guardar los campos del MREP2
    ofs.write(reinterpret_cast<const char*>(&ktree->maxLevel), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&ktree->numberOfNodes), sizeof(size_t));
    ofs.write(reinterpret_cast<const char*>(&ktree->numberOfEdges), sizeof(size_t));

    // Guardar el bitmap BT
    if (ktree->bt) {
        size_t btSize = ktree->bt->getSize();
        ofs.write(reinterpret_cast<const char*>(&btSize), sizeof(size_t));  // Guardar tamaño del bitmap
        ktree->bt->save(ofs);  // Usar el método de guardado de la clase
    } else {
        size_t btSize = 0;
        ofs.write(reinterpret_cast<const char*>(&btSize), sizeof(size_t));  // Guardar tamaño cero si no existe
    }

    // Guardar el bitmap BN
    if (ktree->bn) {
        size_t bnSize = ktree->bn->getSize();
        ofs.write(reinterpret_cast<const char*>(&bnSize), sizeof(size_t));  // Guardar tamaño del bitmap
        ktree->bn->save(ofs);  // Usar el método de guardado de la clase
    } else {
        size_t bnSize = 0;
        ofs.write(reinterpret_cast<const char*>(&bnSize), sizeof(size_t));  // Guardar tamaño cero si no existe
    }

    // Guardar el bitmap BL
    if (ktree->bl) {
        size_t blSize = ktree->bl->getSize();
        ofs.write(reinterpret_cast<const char*>(&blSize), sizeof(size_t));  // Guardar tamaño del bitmap
        ktree->bl->save(ofs);  // Usar el método de guardado de la clase
    } else {
        size_t blSize = 0;
        ofs.write(reinterpret_cast<const char*>(&blSize), sizeof(size_t));  // Guardar tamaño cero si no existe
    }

    // Guardar la tabla de divisiones de nivel
    ofs.write(reinterpret_cast<const char*>(ktree->div_level_table), sizeof(unsigned long long int) * (ktree->maxLevel));

    return 0;
}
int loadMREP2FromFile(MREP2 **ktree, std::ifstream &ifs) {
    *ktree = (MREP2 *)malloc(sizeof(MREP2));
    if (!*ktree) {
        perror("Error al asignar memoria para MREP2");
        return -1;
    }

    // Leer los campos básicos
    if (!ifs.read(reinterpret_cast<char*>(&(*ktree)->maxLevel), sizeof(int))) {
        perror("Error al leer maxLevel");
        free(*ktree);
        return -1;
    }
    if (!ifs.read(reinterpret_cast<char*>(&(*ktree)->numberOfNodes), sizeof(size_t))) {
        perror("Error al leer numberOfNodes");
        free(*ktree);
        return -1;
    }
    if (!ifs.read(reinterpret_cast<char*>(&(*ktree)->numberOfEdges), sizeof(size_t))) {
        perror("Error al leer numberOfEdges");
        free(*ktree);
        return -1;
    }

    // Cargar el bitmap BT
    size_t btSize;
    if (!ifs.read(reinterpret_cast<char*>(&btSize), sizeof(size_t))) {
        perror("Error al leer el tamaño del bitmap BT");
        free(*ktree);
        return -1;
    }
    if (btSize > 0) {
        (*ktree)->bt = BitSequence::load(ifs);  // Asegúrate de que esta función maneje la memoria correctamente
    } else {
        (*ktree)->bt = nullptr;
    }

    // Cargar el bitmap BN
    size_t bnSize;
    if (!ifs.read(reinterpret_cast<char*>(&bnSize), sizeof(size_t))) {
        perror("Error al leer el tamaño del bitmap BN");
        free(*ktree);
        return -1;
    }
    if (bnSize > 0) {
        (*ktree)->bn = BitSequence::load(ifs);
    } else {
        (*ktree)->bn = nullptr;
    }

    // Cargar el bitmap BL
    size_t blSize;
    if (!ifs.read(reinterpret_cast<char*>(&blSize), sizeof(size_t))) {
        perror("Error al leer el tamaño del bitmap BL");
        free(*ktree);
        return -1;
    }
    if (blSize > 0) {
        (*ktree)->bl = BitSequence::load(ifs);
    } else {
        (*ktree)->bl = nullptr;
    }

    // Cargar la tabla de divisiones de nivel
    (*ktree)->div_level_table = (unsigned long long int *)malloc(sizeof(unsigned long long int) * ((*ktree)->maxLevel));
    if (!(*ktree)->div_level_table) {
        perror("Error al asignar memoria para div_level_table");
        free(*ktree);
        return -1;
    }
    if (!ifs.read(reinterpret_cast<char*>((*ktree)->div_level_table), sizeof(unsigned long long int) * ((*ktree)->maxLevel ))) {
        perror("Error al leer div_level_table");
        free((*ktree)->div_level_table);
        free(*ktree);
        return -1;
    }

    return 0;
}

// Guardar Snapshot en archivo
int saveSnapshotToFile(Snapshot *snapshot, const char *filename) {
    if (!snapshot || !filename) return -1;

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        perror("Error abriendo el archivo para guardar el Snapshot");
        return -1;
    }

    // Guardar el número total de objetos
    ofs.write(reinterpret_cast<const char*>(&snapshot->totObj), sizeof(uint));

    // Guardar el MREP2 (árbol k2tree)
    if (saveMREP2ToFile(snapshot->ktree, ofs) != 0) {
        ofs.close();
        return -1;
    }

    // Guardar el LabelsMap
    // Suponiendo que tienes una función para guardar LabelsMap
    // if (saveLabelsMapToFile(snapshot->labels, ofs) != 0) {
    //     ofs.close();
    //     return -1;
    // }

    ofs.close();
    return 0;
}
int loadSnapshotFromFile(Snapshot **snapshot, const char *filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        perror("Error abriendo el archivo para cargar");
        return -1;
    }

    *snapshot = (Snapshot *)malloc(sizeof(Snapshot));
    if (!*snapshot) {
        perror("Error al asignar memoria para Snapshot");
        return -1;
    }

    // Leer el total de objetos
    if (!ifs.read(reinterpret_cast<char*>(&(*snapshot)->totObj), sizeof(uint))) {
        perror("Error al leer totObj");
        free(*snapshot);
        return -1;
    }

    // Leer el MREP2
    if (loadMREP2FromFile(&(*snapshot)->ktree, ifs) != 0) {
        perror("Error al cargar MREP2 desde archivo");
        free(*snapshot);
        return -1;
    }

    // Leer el LabelsMap
    (*snapshot)->labels = NULL;

    return 0;
}




void firstLeaf(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
               uint dp, uint dq, int x, int l, uint &n, Point &p) {
    MREP2 *rep = snap->ktree;
    uint i = 0, j, leaf;
    uint y, p1new, p2new, q1new, q2new;
    unsigned long int divlevel;
    if (l == rep->maxLevel) {
        //recorrido por el bitarray leavesInf
        leaf = x + i * p1;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                leaf = x + j;
                if (rep->bl->access(leaf)) {
                    //Point p;
                    p.setX(dp + i);
                    p.setY(dq + j);
                    return;
                    //dp + i => posición X del objeto econtrado
                    //dq + j => posición Y del objeto encontrado.
                    //snap->labels->getObjects(rep->bl->rank1(leaf), dp + i,dq + j, Oid, X, Y, n);

                }
            }
            leaf += K;
        }

    }

    if ((l == rep->maxLevel - 1)
        && (rep->bn->access(x - rep->bt->getLength()))) {
        //recorrido por el bitarray bn
        y = (rep->bn->rank1(x - rep->bt->getLength()) - 1) * K * K;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                firstLeaf(snap, 0, 0, 0, 0, dp + i, dq + j,
                          y + K * i + j, l + 1, n, p);
            }
        }

    }
    if ((x == -1) || ((l < rep->maxLevel - 1) && (rep->bt->access(x)))) {
        //recorrido por el bitarray bt
        y = (x == -1) ? 0 : rep->bt->rank1(x) * K * K;
        divlevel = rep->div_level_table[l + 1];
        for (i = p1 / divlevel; i <= p2 / divlevel; i++) {
            p1new = 0;
            if (i == p1 / divlevel)
                p1new = p1 % divlevel;
            p2new = divlevel - 1;
            if (i == p2 / divlevel)
                p2new = p2 % divlevel;
            for (j = q1 / divlevel; j <= q2 / divlevel; j++) {
                q1new = 0;
                if (j == q1 / divlevel)
                    q1new = q1 % divlevel;
                q2new = divlevel - 1;
                if (j == q2 / divlevel)
                    q2new = q2 % divlevel;
                firstLeaf(snap, p1new, p2new, q1new, q2new,
                          dp + divlevel * i, dq + divlevel * j, y + K * i + j,
                          l + 1, n, p);
            }
        }

    }
}

//arreglo donde la posición 0 tiene el total de elementos (n) y comienza en 1 y
//termina en n. n es un parámetro de salida.
void rangeQuery(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
                uint *&Oid, uint *&X, uint *&Y, uint &n) {
    Oid = new uint[snap->totObj]();
    X = new uint[snap->totObj]();
    Y = new uint[snap->totObj]();
    n = 0;
    recursiveGetRank(snap, p1, p2, q1, q2, 0, 0, -1, -1, Oid, X, Y, n);
}

std::stack<uint> *findPath(Snapshot *snap, uint oid) {
    stack<uint> *path = new stack<uint>();
    size_t p = snap->labels->getListID(oid);
    cout << "Prueba" << oid << endl;
    if (p > snap->ktree->bl->countOnes()) {
        //significa que el objeto no está en el snapshot.
        return path;
    }
    size_t x = snap->ktree->bl->select1(p);
    uint i = x % (K * K);                    //posición como hijo en bl
    path->push(i);
    //posicion del padre de x. en bn
    x = snap->ktree->bn->select1((x / (K * K)) + 1);
    x = x + snap->ktree->bt->getLength();
    while (x > 0) {
        i = x % (K * K);                    //posición como hijo
        path->push(i);
        //posicion del padre de x.
        x = (x < (K * K) ? 0 : snap->ktree->bt->select1((x / (K * K))));
    }
    return path;
}

Point *getObjectPos(Snapshot *snap, uint oid) {
    if (oid > snap->totObj) {
        return NULL;
    }
    uint x1 = 0, y1 = 0, x2 = (snap->ktree->numberOfNodes - 1), y2 = x2;
    stack<uint> *path = findPath(snap, oid);
    uint i, colum, row, piv_x, piv_y;
    while (!path->empty()) {
        i = path->top();
        colum = i % K;
        row = i / K;
        piv_x = ((x2 - x1) + 1) / K;
        piv_y = ((y2 - y1) + 1) / K;
        x2 = piv_x * (row + 1) + x1 - 1;
        x1 = x2 - piv_x + 1;
        y2 = piv_y * (colum + 1) + y1 - 1;
        y1 = y2 - piv_y + 1;
        path->pop();
    }
    delete path;
    path = NULL;
    return new Point(x1, y1);
}

typedef struct sArea {
    unsigned long int x1, x2, y1, y2;
} SpatialArea;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * element Queue
 */
typedef struct sElementQueue {
    long int sTree;
    SpatialArea area;
    double priority; //la distancia...
} ElementQueue;
void printElementQueue(ElementQueue e) {
    printf("ElementQueue:\n");
    printf("  sTree: %ld\n", e.sTree);
    printf("  Area: [x1=%lu, x2=%lu, y1=%lu, y2=%lu]\n",
           e.area.x1, e.area.x2, e.area.y1, e.area.y2);
    printf("  Priority (distancia): %.6f\n", e.priority);
}
typedef struct sCandidateQueue {
    long int oid;
    long int x;
    long int y;
    double priority;
} CandidateElement;


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Max Heap comparator
 */
struct maxHeapComparator {
    bool operator()(CandidateElement i, CandidateElement j) {
        return i.priority < j.priority;
    }
};

struct maxHeapComparatorHD {
    bool operator()(ElementQueue i, ElementQueue j) {
        return i.priority < j.priority;
    }
};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Min Heap comparator
 */
struct minHeapComparator {
    bool operator()(ElementQueue i, ElementQueue j) {
        return i.priority > j.priority;
    }
};

typedef struct sPriorityQueue {
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pPila;
    int bandera;
} PriorityQueue;
typedef struct sHdQueue {
    PriorityQueue pPila;
    ElementQueue element;
    int bandera;
} HdQueue;

double maxDist(Point p, SpatialArea a) {
    long int dx1 = (p.getX() - a.x1);
    dx1 = dx1 >= 0 ? dx1 : -dx1;

    long int dx2 = (p.getX() - a.x2);
    dx2 = dx2 >= 0 ? dx2 : -dx2;

    long int dy1 = p.getY() - a.y1;
    dy1 = dy1 >= 0 ? dy1 : -dy1;

    long int dy2 = p.getY() - a.y2;
    dy2 = dy2 >= 0 ? dy2 : -dy2;

    long int dx = max(dx1, dx2);
    long int dy = max(dy1, dy2);
    return sqrt((dx * dx) + (dy * dy));
    //return (dx * dx) + (dy * dy);
}

double maxDistSinSqrt(Point p, SpatialArea a) {
    long int dx1 = (p.getX() - a.x1);
    dx1 = dx1 >= 0 ? dx1 : -dx1;

    long int dx2 = (p.getX() - a.x2);
    dx2 = dx2 >= 0 ? dx2 : -dx2;

    long int dy1 = p.getY() - a.y1;
    dy1 = dy1 >= 0 ? dy1 : -dy1;

    long int dy2 = p.getY() - a.y2;
    dy2 = dy2 >= 0 ? dy2 : -dy2;

    long int dx = max(dx1, dx2);
    long int dy = max(dy1, dy2);
    //return sqrt((dx * dx) + (dy * dy));
    
    return (dx * dx) + (dy * dy);
}

//Funcion que calcula la mayor distancia entre areas
double maxDistAreas(SpatialArea s, SpatialArea b) {
    Point p;
    double d, d2, d3, d4;
    //Esto debemos optimizarlo
    //Obtenemos el primer vertice
    p.setX(s.x1);
    p.setY(s.y1);
    d = maxDistSinSqrt(p, b);
    //Obtenemos el segundo vertice
    p.setX(s.x1);
    p.setY(s.y2);
    d2 = maxDistSinSqrt(p, b);
    if (d2 > d) {
        d = d2;
    }
    //Obtenemos el tercer vertice
    p.setX(s.x2);
    p.setY(s.y1);
    d3 = maxDistSinSqrt(p, b);
    if (d3 > d) {
        d = d3;
    }
    //Obtenemos el cuarto vertice
    p.setX(s.x2);
    p.setY(s.y2);
    d4 = maxDistSinSqrt(p, b);
    if (d4 > d) {
        d = d4;
    }
    return d;
}

double minDist(Point p, SpatialArea a) {
    long int x = p.getX();
    long int y = p.getY();
    if (x < a.x1) {
        //hay 3 casos o bien está dentro del rango y1,y2 o fuera.
        if (y < a.y1) {
            long int dx = a.x1 - x;
            long int dy = a.y1 - y;
            return sqrt((dx * dx) + (dy * dy));
            //return (dx * dx) + (dy * dy);
        } else if (y <= a.y2) {
            return double(a.x1 - x);
        } else {
            long int dx = a.x1 - x;
            long int dy = y - a.y2;
            return sqrt((dx * dx) + (dy * dy));
            //return (dx * dx) + (dy * dy);
        }
    } else if (x <= a.x2) {
        if (y < a.y1) {
            return double(a.y1 - y);
        } else if (y <= a.y2) {
            return min(min((x - a.x1), (a.x2 - x)), min((y - a.y1), (a.y2 - y)));
        } else {
            return double(y - a.y2);
        }
    } else {
        if (y < a.y1) {
            long int dx = x - a.x2;
            long int dy = a.y1 - y;
            return sqrt((dx * dx) + (dy * dy));
            //return (dx * dx) + (dy * dy);
        } else if (y <= a.y2) {
            return double(x - a.x2);
        } else {
            long int dx = x - a.x2;
            long int dy = y - a.y2;
            return sqrt((dx * dx) + (dy * dy));
            //return (dx * dx) + (dy * dy);
        }
    }
    return 0.0;
}

double minDistSinSqrt(Point p, SpatialArea a) {
    long int x = p.getX();
    long int y = p.getY();
    if (x < a.x1) {
        //hay 3 casos o bien está dentro del rango y1,y2 o fuera.
        if (y < a.y1) {
            long int dx = a.x1 - x;
            long int dy = a.y1 - y;
           
            return (dx * dx) + (dy * dy);
            //return (dx * dx) + (dy * dy);
        } else if (y <= a.y2) {
            return double((a.x1 - x) * (a.x1 - x));
        } else {
            long int dx = a.x1 - x;
            long int dy = y - a.y2;
            return (dx * dx) + (dy * dy);
            //return (dx * dx) + (dy * dy);
        }
    } else if (x <= a.x2) {
        if (y < a.y1) {
           
            return double((a.y1 - y) * (a.y1 - y));
        } else if (y <= a.y2) {
            //return min(min((x - a.x1), (a.x2 - x)), min((y - a.y1), (a.y2 - y)));
            double aux = min(min((x - a.x1), (a.x2 - x)), min((y - a.y1), (a.y2 - y)));
            return aux * aux;
        } else {
            return double((y - a.y2) * (y - a.y2));
        }
    } else {
        if (y < a.y1) {
            long int dx = x - a.x2;
            long int dy = a.y1 - y;
            
            return (dx * dx) + (dy * dy);
            //return (dx * dx) + (dy * dy);
        } else if (y <= a.y2) {
            
            return double((x - a.x2) * (x - a.x2));
        } else {
            long int dx = x - a.x2;
            long int dy = y - a.y2;
            
            //return sqrt((dx * dx) + (dy * dy));
            return (dx * dx) + (dy * dy);
        }
    }

    return 0.0;
}

double minDistSinSqrt_(Point p, SpatialArea a) {
    long int x = p.getX();//Obtenemos la Coordenada
    long int y = p.getY();//Obtenemos la Coordenada
    if (x < a.x1) {//Si esta fuera del rango en X
        //hay 3 casos o bien está dentro del rango y1,y2 o fuera.
        if (y < a.y1) { //Caso 1, esta fuera del rango en x e y
            long int dx = a.x1 - x;
            long int dy = a.y1 - y;
            //return sqrt((dx * dx) + (dy * dy));
            return (dx * dx) + (dy * dy);//Sacamos la raiz cuadrada
        } else if (y <= a.y2) {
            //return double(a.x1 - x);
            long int dx = a.x1 - x;
            long int dy = a.y1 - y; //y1-y va ser siempre 0, las distancia mas corta es perpendicular a y
            return (dx * dx) + (dy * dy);//Sacamos la raiz cuadrada
        } else {
            long int dx = a.x1 - x;
            long int dy = y - a.y2;
            //return sqrt((dx * dx) + (dy * dy));
            return (dx * dx) + (dy * dy);
        }
    } else if (x <= a.x2) {
        if (y < a.y1) {
            long int dy = a.y1 - y;
            long int dx = a.x2 - x;
            return (dx * dx) + (dy * dy);
            //return double(a.y1 - y);
        } else if (y <= a.y2) {
            //return min(min((x - a.x1), (a.x2 - x)), min((y - a.y1), (a.y2 - y)));
            double aux = min(min((x - a.x1), (a.x2 - x)), min((y - a.y1), (a.y2 - y)));
            return aux * aux;
        } else {
            long int dy = y - a.y2;
            long int dx = a.x2 - x;//Siempre sera 0 porque es perpendicular al eje x
            return (dx * dx) + (dy * dy);
            //return double(y - a.y2);
        }
    } else {
        if (y < a.y1) {
            long int dx = x - a.x2;
            long int dy = a.y1 - y;
            //return sqrt((dx * dx) + (dy * dy));
            return (dx * dx) + (dy * dy);
        } else if (y <= a.y2) {
            long int dx = x - a.x2;
            long int dy = a.y1 - y;
            return (dx * dx) + (dy * dy);
            //return double(x - a.x2);
        } else {
            long int dx = x - a.x2;
            long int dy = y - a.y2;
            //return sqrt((dx * dx) + (dy * dy));
            return (dx * dx) + (dy * dy);
        }
    }
    return 0.0;
}

inline int isLeaf(MREP2 *mrep, long int index) {
    //index está mapeado como si la implementación fuera un único bitmap
    //con la finalidad de encapsular las diferentes implementaciones existentes
    //del k^2-tree
    //verdadero si es una hoja de último nivel sino falso.
    //si index es -1 entonces es la raíz
    return (index != -1)
           && ((mrep->bt->getLength() + mrep->bn->getLength()) <= index);
}

inline int posInLeaf(MREP2 *mrep, long int index) {
    return index - (mrep->bt->getLength() + mrep->bn->getLength());
}

inline bool esCero(MREP2 *mrep, long int x) {
    long int btLen = mrep->bt->getLength();
    long int bnLen = mrep->bn->getLength();
    if (x == -1) {
        return false;
    } else if (x < btLen) {
        return mrep->bt->access(x) == 0;
    } else if (x < (btLen + bnLen)) {
        return mrep->bn->access(x - btLen) == 0;
    } else {
        //es una hoja de último nivel.
        return mrep->bl->access(x - (btLen + bnLen)) == 0;
    }
}

inline long int firstChild(MREP2 *mrep, long int x) {
    long int btLen = mrep->bt->getLength();
    long int bnLen = mrep->bn->getLength();
    long int btOnes = mrep->bt->countOnes();
    if (x == -1) {
        return 0;
    } else if (x < btLen) {
        return (mrep->bt->access(x) == 0) ? -1 : mrep->bt->rank1(x) * K * K;
    } else if (x < btLen + bnLen) {
        /*		cout << endl << "btOnes:" << btOnes << " x in bt:" << x - btLen
         << " rank1 de x en bt:" << mrep->bn->rank1(x - btLen) << endl;
         cout << ((btOnes + (mrep->bn->rank1(x - btLen))) * K * K) << endl;
         */
        return (mrep->bn->access(x - btLen) == 0) ?
               //error: -1 : (btLen + (mrep->bn->rank1(x - btLen) * K * K));
               -1 : ((btOnes + (mrep->bn->rank1(x - btLen))) * K * K);
    } else {
        //es una hoja de último nivel.
        return -1;
    }
}

//Esto está implementada para un K=2, pero hay que hacerlo para cualquier
//valor de K.
inline SpatialArea getSubArea(SpatialArea a, int i) {
    SpatialArea resp;
    switch (i) {
        case 0:
            resp.x1 = a.x1;
            resp.x2 = (a.x1 + a.x2) / 2;
            resp.y1 = a.y1;
            resp.y2 = (a.y1 + a.y2) / 2;
            break;
        case 1:
            resp.x1 = a.x1;
            resp.x2 = (a.x1 + a.x2) / 2;
            resp.y1 = ((a.y1 + a.y2) / 2) + 1;
            resp.y2 = a.y2;

            break;
        case 2:
            resp.x1 = ((a.x1 + a.x2) / 2) + 1;
            resp.x2 = a.x2;
            resp.y1 = a.y1;
            resp.y2 = (a.y1 + a.y2) / 2;

            break;
        case 3:
            resp.x1 = ((a.x1 + a.x2) / 2) + 1;
            resp.x2 = a.x2;
            resp.y1 = ((a.y1 + a.y2) / 2) + 1;
            resp.y2 = a.y2;
            break;
    }
    return resp;

}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/*
 * oid y p son arreglos no nulos de tamaño kn
 */
void kNN(Snapshot *rep, Point q, uint &kn, uint *oid, Point *p) {
    priority_queue<CandidateElement, std::vector<CandidateElement>,
            maxHeapComparator> candidates;
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    ElementQueue e;
    CandidateElement ce;
    e.sTree = 0;
    e.area.x1 = 0;
    e.area.x2 = pow(2, rep->ktree->maxLevel) - 1;
    e.area.y1 = e.area.x1;
    e.area.y2 = e.area.x2;
    e.priority = maxDist(q, e.area);
    long int subTreeIndex;
    double minDis2Child;
    long uint lenCandidates = 0;
    uint numObj = 0;
    uint toAdd = 0;
    uint len = 0;
    uint *resp;
    int rankLeaf;
    pQueue.push(e);
    while (!pQueue.empty()) {
        e = pQueue.top();
        pQueue.pop();
        if (isLeaf(rep->ktree, e.sTree)) {
            //es una hoja de 'ultimo nivel
            lenCandidates = candidates.size();
            if (lenCandidates < kn) {
                numObj = kn - lenCandidates;
                rankLeaf = rep->ktree->bl->rank1(
                        posInLeaf(rep->ktree, e.sTree));
                //trata de recuperar el máximo de objetos para llenar la cola
                //de candidatos.  Puede que no exista más de un objeto en la
                //celda, o bien el número sea menor a los espacios disponibles,
                //en este caso numObj despúes de la llamada tendrá
                //el número real de objetos recuperados
                resp = rep->labels->getUpToKObjects(rankLeaf, numObj);
                for (uint i = 0; i < numObj; i++) {
                    ce.oid = resp[i];
                    //si hubiera información adicional de donde está el objeto
                    //dentro de la celda habría que ajustar la distancia a q
                    //y tambien su x,y.
                    //abría que ajustar la medida.
                    ce.priority = e.priority;
                    //al llegar a la hoja x1==x2 e y1==y2.
                    ce.x = e.area.x1;
                    ce.y = e.area.y1;
                    candidates.push(ce);
                }
                //elimino resp
                delete[] resp;
                resp = NULL;
            } else if (e.priority < (candidates.top().priority)) {
                int i = 0;
                numObj = kn;
                rankLeaf = rep->ktree->bl->rank1(
                        posInLeaf(rep->ktree, e.sTree));
                //aquí no puedo saber cuantos objetos a priory son
                //menores que los actuales candidatos por lo tanto
                //tengo que recuperar Kn objetos, porque en el peor
                //los objetos de la celda reemplazarán a todos los
                //candidatos.  No es necesario recuparar más de Kn
                resp = rep->labels->getUpToKObjects(rankLeaf, numObj);
                do {
                    candidates.pop();
                    ce.oid = resp[i++];
                    ce.priority = e.priority;
                    ce.x = e.area.x1;
                    ce.y = e.area.y1;
                    candidates.push(ce);
                } while (i < numObj && e.priority < (candidates.top().priority));
                delete[] resp;
                resp = NULL;
            }
        } else {
            subTreeIndex = firstChild(rep->ktree, e.sTree);
            if (subTreeIndex != -1) {
                //no es una hoja intermedia, es decir una región sin objetos
                for (int i = 0; i < (K * K); i++) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = maxDist(q, child.area);
                    minDis2Child = minDist(q, child.area);

                    if (!((candidates.size() >= kn)
                          && (minDis2Child >= candidates.top().priority))) {
                        pQueue.push(child);
                    }
                }
            }
        }
    }
    kn = candidates.size();
    //puede ser que existan menos kn que los originalmente pedidos
    //entonces en la respuesta kn serán los realmente obtenidos.
    //lo siguiente es un overhead que se podría evitar si se retorna la cola
    //directamente.
    uint i = kn - 1;
    while (!candidates.empty()) {
        ce = candidates.top();
        oid[i] = ce.oid;
        p[i] = Point(ce.x, ce.y);
        i--;
        candidates.pop();
    }
}

//*************** Vecinos Reversos *********************************
//determina si dist corresponde a la distancia del vecino más cercano a q.
//Inspeccionada por rodrigo
bool isNN(Snapshot *rep, Point q, double dist) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    ElementQueue e;
    e.sTree = -1;
    e.area.x1 = 0;
    e.area.x2 = pow(K, rep->ktree->maxLevel + 1) - 1;
    e.area.y1 = e.area.x1;
    e.area.y2 = e.area.x2;
    e.priority = maxDist(q, e.area);

    long int subTreeIndex;
    double minDis2Child;
    pQueue.push(e);
    while (!pQueue.empty()) {
        e = pQueue.top();
        pQueue.pop();
        if (isLeaf(rep->ktree, e.sTree)) {
            //si es hoja de último nivel
            if ((e.priority < dist)) {
                //encontré otro más cercano
                return false;
            } else {
                //aquí es igual.
                return true;
            }
        } else {
            subTreeIndex = firstChild(rep->ktree, e.sTree);
            //child function return -1 if sTree no have any child
            //no tiene hijo si en la posición e.sTreee hay un 0.

            for (int i = 0; i < (K * K); i++) {
                if (!esCero(rep->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = maxDist(q, child.area);
                    if (child.priority < dist) {
                        return false;
                    }
                    minDis2Child = minDist(q, child.area);
                    if (minDis2Child <= dist) {
                        pQueue.push(child);
                    }
                }
            }
        }
    }
    return false;
}

//Inspeccionada con rodrigo
void intersectX(Point p, Point q, ElementQueue e, int &xi, int &xd) {
    int x1, x2, Ym, Xm;
    Xm = (p.getX() + q.getX()) / 2;
    Ym = (p.getY() + q.getY()) / 2;

    if (p.getX() == q.getX()) {
        //bicectris paralela al eje y.

        if (p.getY() > q.getY()) {
            if (e.area.y1 > Ym) {
                //xi=infinito.
                xi = std::numeric_limits<int>::max();;
            }
        } else {
            if (e.area.y2 > Ym) {
                //xi=infinito.
                xi = std::numeric_limits<int>::max();;
            }

        }

    } else {
        x1 = ((Ym - e.area.y1) * (p.getY() - q.getY())) / (p.getX() - q.getX())
             + Xm;
        x2 = ((Ym - e.area.y2) * (p.getY() - q.getY())) / (p.getX() - q.getX())
             + Xm;
        if (p.getX() > q.getX()) {
            xd = min(max(x1, x2), xd);
        } else {
            xi = max(min(x1, x2), xi);
        }
    }
}

bool isPrunable(ElementQueue e,
                priority_queue<ElementQueue, std::vector<ElementQueue>,
                        minHeapComparator> cand, Point q) {
    ElementQueue c;
    int xi = e.area.x1 - 1;
    int xd = e.area.x2 + 1;
    while (!cand.empty()) {
        c = cand.top();
        cand.pop();
        Point p(c.area.x1, c.area.y1);
        intersectX(p, q, e, xi, xd);
        if (xi + 1 >= xd) {
            return true;
        }
    }
    return false;
}

void displayQueueAction(char *tipo, ElementQueue e) {
    cout << tipo << "(Area[" << e.area.x1 << "--" << e.area.x2 << " : "
         << e.area.y1 << "--" << e.area.y2 << "], priority: " << e.priority
         << ", sTree: " << e.sTree << ")" << endl;
}

void RNN(Snapshot *rep, Point q, uint &n, uint *&oid, Point &out) {
    //pre:asumimos que el k2-tree tiene puntos.

    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> cand;
    ElementQueue e;
    e.sTree = -1;
    e.area.x1 = 0;
//el 2 es porque k = 2, reemplazar por K.
    e.area.x2 = pow(K, rep->ktree->maxLevel + 1) - 1;
    e.area.y1 = e.area.x1;
    e.area.y2 = e.area.x2;
    e.priority = maxDist(q, e.area);

    long int subTreeIndex;
    double minDis2Child;

    pQueue.push(e);
    displayQueueAction("pQueue.push", e);
    while (!pQueue.empty()) {
        e = pQueue.top();
        pQueue.pop();
        displayQueueAction("	pQueue.pop", e);
        if (isLeaf(rep->ktree, e.sTree)) {
            if (!isPrunable(e, cand, q)) {
                cand.push(e);
                displayQueueAction("			----> cand.push", e);
            }
        } else {
            subTreeIndex = firstChild(rep->ktree, e.sTree);
            //child function return -1 if sTree no have any child
            //no tiene hijo si en la posición e.sTreee hay un 0.

            for (int i = 0; i < (K * K); i++) {
                if (!esCero(rep->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = minDist(q, child.area);

                    if (!isPrunable(child, cand, q)) {
                        pQueue.push(child);
                        displayQueueAction("pQueue.push", child);
                    }
                }
            }
        }
    }
//aquí hay que seleccionar los puntos que son parte del
//resultado.
    n = 0;

    /**
     * TODO: AQUÍ TENGO QUE REVISAR SI FUNCIONA BIEN LA CLASE PUNTO
     * ¿SE NECESITA DEFINIR UN COPY CONSTRUCTOR O BIEN = BASTA EN ESTE CASO?
     */
    while (!cand.empty()) {
        e = cand.top();
        cand.pop();
        out.setX(e.area.x1);
        out.setY(e.area.y1);
        if (isNN(rep, out, e.priority)) {

            //sacar la lista de OID podría quedar fuera de la operación
            int rankLeaf = rep->ktree->bl->rank1(
                    posInLeaf(rep->ktree, e.sTree));
            uint len;
            uint *id = rep->labels->getObjects(rankLeaf, len);
            oid = new uint[len];
            out.setX(e.area.x1);
            out.setY(e.area.y1);;
            for (int i = 0; i < len; i++) {
                oid[n++] = id[i];
            }
        }
    }

}

//************************** HAUSDORF DISTANCE ************************************************
//busca encontrar mayor de las menores distancias. Para ello pasa como parámetro max que
//es la mayor de las menores distancias encontradas hasta ahora.
/*void nnMaxOptSinSqrt(Snapshot * rep, Point q, double &max, Point &pnn, double d) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    double c = std::numeric_limits<double>::max();    //
    //double aux = 0.0;
    int bandera = 0;
    //int inicio = 1;
    //int bandera2 = 0;
    ElementQueue e;
    Point cnn;
    e.sTree = -1;
    e.area.x1 = 0;
//el 2 es porque k = 2, reemplazar por K.
    e.area.x2 = pow(K, rep->ktree->maxLevel+1) - 1;
    //cout << "X2: " << e.area.x2 << endl;
    e.area.y1 = e.area.x1;
    e.area.y2 = e.area.x2;
    //e.priority = maxDist(q, e.area);
    e.priority = d;
    //Se calcula la maxima distancia hacia los vertices
    //cout << "Distancia Inicial: " << e.priority << endl;
    //cout << "Entra: " << d << endl;
    //cout << "Max: " << max << endl;
    long int subTreeIndex;
    double minDis2Child;
    pQueue.push(e);
    while (!pQueue.empty()) {
        e = pQueue.top();
        pQueue.pop();
        if (isLeaf(rep->ktree, e.sTree)) {
            //si es hoja de último nivel
            if ((e.priority <= max )) {
                //cout << "RETORNO 1" << endl;
                //return max;
                return;
            }else{
				if ((e.priority < d ) && max != 0.0) {
					cnn.setX(e.area.x1);
					cnn.setY(e.area.y1);
					//c = e.priority;
					d = e.priority;

				}
				else{
					c = e.priority;
					//bandera2 = 0;

				}
            }
            //cout << "HOJA?: " << e.priority << endl;
        } else {
            subTreeIndex = firstChild(rep->ktree, e.sTree);
            //child function return -1 if sTree no have any child
            //no tiene hijo si en la posición e.sTreee hay un 0.

            for (int i = 0; i < (K * K); i++) {
                if (!esCero(rep->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = maxDistSinSqrt(q, child.area);

                    if (child.priority <= max) {
                        //return max;
                        //cout << "RETORNO 2" << endl;
                        return;
                    }else{

						minDis2Child = minDistSinSqrt(q, child.area);

						if ((minDis2Child < d) && (max != 0.0)) {
							//cout << "Menor Distancia Correcta: " << minDis2Child << endl;
							bandera = 1;
							pQueue.push(child);
							//inicio = 0;
						}
						else{

							//minDis2Child = minDist(q, child.area);
							if (minDis2Child < c) {
									pQueue.push(child);
							}
						}
                    }

                }
            }
        }
    }
    if(bandera == 1 && max != 0.0){
        pnn = cnn;
        //cout << max << endl;
        max = d;
    }
    else{
    	max = c;
    }
    //cout << "Sale: " << max << endl;

}*/
/*FUNCION QUE NO UTILIZA PNN o CMIN*/
/**************************************************************/
void nnMax(Snapshot *rep, Point q, double &max) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    double c = std::numeric_limits<double>::max();    //
    ElementQueue e;
    e.sTree = -1;
    e.area.x1 = 0;
//el 2 es porque k = 2, reemplazar por K.
    e.area.x2 = pow(K, rep->ktree->maxLevel + 1) - 1;
    e.area.y1 = e.area.x1;
    e.area.y2 = e.area.x2;
    e.priority = maxDistSinSqrt(q, e.area);//Se calcula la maxima distancia hacia los vertices
    long int subTreeIndex;
    double minDis2Child;
    pQueue.push(e);
    while (!pQueue.empty()) {
        e = pQueue.top();
        pQueue.pop();
        if (isLeaf(rep->ktree, e.sTree)) {
            //si es hoja de último nivel
            if ((e.priority <= max)) {

                //return max;
                return;
            } else {
                if (e.priority < c) {
                    c = e.priority;
                }
            }
            //cout << "HOJA?: " << e.priority << endl;
        } else {
            subTreeIndex = firstChild(rep->ktree, e.sTree);
            //child function return -1 if sTree no have any child
            //no tiene hijo si en la posición e.sTreee hay un 0.

            for (int i = 0; i < (K * K); i++) {
                if (!esCero(rep->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = maxDistSinSqrt(q, child.area);
                    if (child.priority <= max) {
                        //return max;
                        return;
                    }
                    minDis2Child = minDistSinSqrt(q, child.area);
                    if (minDis2Child < c) {
                        pQueue.push(child);
                    }
                }
            }
        }
    }
    max = c;
    //cout << "CMAX: " << sqrt(max) << endl;
}

/*FUNCION QUE ESCANEA COLAS ORDENADAS DE MENOR A MAYOR (MINHEAP)
/**************************************************************/
void nnMaxQueue(PriorityQueue &pfilaB, Snapshot *rep, Point q, double &max) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> auxQueue;
    double c = std::numeric_limits<double>::max();    //
    /*auxQueue = pfilaB.pPila;
    ElementQueue aux1 = auxQueue.top();*/
    //cout << sqrt(aux1.priority) << endl;
    //pQueue = pq;
    //auxQueue = pq.pPila;
    pQueue = pfilaB.pPila;
    ElementQueue aux = pQueue.top();
    //cout << "Priority" << aux.priority << "Stree" << aux.sTree << endl;
    ElementQueue e;
    e.sTree = aux.sTree;
    e.area.x1 = aux.area.x1;
    //el 2 es porque k = 2, reemplazar por K.
    e.area.x2 = aux.area.x2;
    e.area.y1 = aux.area.y1;
    e.area.y2 = aux.area.y2;
    e.priority = maxDistSinSqrt(q, e.area);//Se calcula la maxima distancia hacia los vertices
    //cout << "Priority" << sqrt(e.priority) << " Stree"<< e.sTree << endl;
    long int subTreeIndex;
    double minDis2Child;
    pQueue.push(e);

    while (!pQueue.empty()) {
        cout << "PASO POR AQUI" << sqrt(e.priority) << endl;
        e = pQueue.top();
        pQueue.pop();
        if (isLeaf(rep->ktree, e.sTree)) {
            //si es hoja de último nivel
            if ((e.priority <= max)) {

                //return max;
                return;
            } else {
                if (e.priority < c) {
                    c = e.priority;
                }
            }
            //cout << "HOJA?: " << e.priority << endl;
        } else {
            subTreeIndex = firstChild(rep->ktree, e.sTree);
            //child function return -1 if sTree no have any child
            //no tiene hijo si en la posición e.sTreee hay un 0.

            for (int i = 0; i < (K * K); i++) {
                if (!esCero(rep->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = maxDistSinSqrt(q, child.area);
                    if (child.priority <= max) {
                        //return max;
                        return;
                    }
                    minDis2Child = minDistSinSqrt(q, child.area);
                    if (minDis2Child < c) {
                        pQueue.push(child);
                    }
                    /*if (minDis2Child < c) {
                        pQueue.push(child);
                    }*/
                }
            }
        }
    }
    max = c;
    cout << max << endl;
    return;
}
//*********************************************************************************************
/*FUNCION QUE UTILIZA PNN o CMIN*/
/**************************************************************/
void nnMaxOpt(Snapshot *rep, Point q, double &max, Point &pnn, double d) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    double c = std::numeric_limits<double>::max();    //
    ElementQueue e;
    Point cnn;
    e.sTree = -1;
    e.area.x1 = 0;
    e.area.x2 = pow(K, rep->ktree->maxLevel + 1) - 1;
    e.area.y1 = e.area.x1;
    e.area.y2 = e.area.x2;
    //e.priority = d;
    e.priority = maxDistSinSqrt(q, e.area);
    long int subTreeIndex;
    double minDis2Child;
    pQueue.push(e);
    while (!pQueue.empty()) {
        e = pQueue.top();
        pQueue.pop();
        if (isLeaf(rep->ktree, e.sTree)) {
            //si es hoja de último nivel
            if ((e.priority <= max)) {
                return;
            } else {
                if ((e.priority < d) && max != 0.0) { //Si Cmax es 0.0, tira valores errones
                    cnn.setX(e.area.x1);
                    cnn.setY(e.area.y1);
                    pnn = cnn;
                    d = e.priority;

                } else {
                    c = e.priority;
                }
            }
        } else {
            subTreeIndex = firstChild(rep->ktree, e.sTree);
            //child function return -1 if sTree no have any child
            //no tiene hijo si en la posición e.sTreee hay un 0.
            for (int i = 0; i < (K * K); i++) {
                if (!esCero(rep->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = maxDistSinSqrt(q, child.area);
                    if (child.priority <= max) {
                        return;
                    } else {
                        minDis2Child = minDistSinSqrt(q, child.area);
                        if ((minDis2Child <= d) && (max != 0.0)) {
                            pQueue.push(child);
                        } else {
                            if (minDis2Child < c) {
                                pQueue.push(child);
                            }
                        }
                    }

                }
            }
        }
    }
    if (max != 0.0) {
        //pnn = cnn;
        max = d;
    } else {
        max = c;
    }
    /*cout << "CMAX: " << sqrt(max) << endl;
    cout << "D: " << sqrt(d) << endl;
    cout << "C: " << sqrt(c) << endl;*/
}

//*********************************************************************************************
//NO SE UTILIZA
void nnMaxSuperOpt(Snapshot *rep, Point q, double &max, Point &pnn, double d) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    double c = std::numeric_limits<double>::max();    //
    //double aux = 0.0;
    int bandera = 0;
    //int inicio = 1;
    //int bandera2 = 0;
    ElementQueue e;
    Point cnn;
    e.sTree = -1;
    e.area.x1 = 0;
    e.area.x2 = pow(K, rep->ktree->maxLevel + 1) - 1;
    e.area.y1 = e.area.x1;
    e.area.y2 = e.area.x2;
    e.priority = d;
    //Se calcula la maxima distancia hacia los vertices
    //cout << "Distancia Inicial: " << e.priority << endl;
    //cout << "Entra: " << d << endl;
    //cout << "Max: " << max << endl;
    long int subTreeIndex;
    double minDis2Child;
    pQueue.push(e);
    while (!pQueue.empty()) {
        e = pQueue.top();
        pQueue.pop();
        if (isLeaf(rep->ktree, e.sTree)) {
            //si es hoja de último nivel
            if ((e.priority <= max)) {
                //cout << "RETORNO 1" << endl;
                //return max;
                return;
            } else {
                if (e.priority < d) {
                    cnn.setX(e.area.x1);
                    cnn.setY(e.area.y1);
                    //c = e.priority;
                    d = e.priority;

                }
                /*else{
                    //cnn.setX(e.area.x1);
                    //cnn.setY(e.area.y1);
                    c = e.priority;
                    //bandera2 = 0;

                }*/
            }
            //cout << "HOJA?: " << e.priority << endl;
        } else {
            subTreeIndex = firstChild(rep->ktree, e.sTree);
            //child function return -1 if sTree no have any child
            //no tiene hijo si en la posición e.sTreee hay un 0.

            for (int i = 0; i < (K * K); i++) {
                if (!esCero(rep->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = maxDistSinSqrt(q, child.area);

                    if (child.priority <= max) {
                        //return max;
                        //cout << "RETORNO 2" << endl;
                        return;
                    } else {

                        minDis2Child = minDistSinSqrt(q, child.area);

                        if (minDis2Child < d) {
                            //cout << "Menor Distancia Correcta: " << minDis2Child << endl;
                            bandera = 1;
                            pQueue.push(child);
                            //inicio = 0;
                        }
                        /*else{

                            if (minDis2Child < c) {
                                    pQueue.push(child);
                            }
                        }*/
                    }

                }
            }
        }
    }
    if (bandera == 1) {
        pnn = cnn;
        //cout << max << endl;
        max = d;
    }
    //else{
    //	max = c;
    //}
    //cout << "CMAX: " << max << endl;

}
//*********************************************************************************************
/*FUNCION QUE NO UTILIZA PODAS POR LA VARIABLE CMAX, SOLO UTILIZA LAS PODAS PROPIAS DEL K2-TREE*/
/**************************************************************/
double nnMaxSinCmax(Snapshot *rep, Point q) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    double c = std::numeric_limits<double>::max();    //
    ElementQueue e;
    //Point cnn;
    e.sTree = -1;
    e.area.x1 = 0;
    e.area.x2 = pow(K, rep->ktree->maxLevel + 1) - 1;
    e.area.y1 = e.area.x1;
    e.area.y2 = e.area.x2;
    e.priority = maxDistSinSqrt(q, e.area);
    //cout << "INICIO: " << e.priority << endl;
    //e.priority = d;
    //Se calcula la maxima distancia hacia los vertices
    long int subTreeIndex;
    double minDis2Child;
    pQueue.push(e);
    while (!pQueue.empty()) {
        e = pQueue.top();
        pQueue.pop();
        if (isLeaf(rep->ktree, e.sTree)) {
            //si es hoja de último nivel
            /*if ((e.priority <= max)) {
                //cout << "RETORNO 1" << endl;
                //return max;
                return;
            } else {
                pnn.setX(e.area.x1);
                pnn.setY(e.area.y1);*/
            if (e.priority < c) {
                c = e.priority;
            }
            //cout << "HOJA?: " << e.priority << endl;
        } else {
            subTreeIndex = firstChild(rep->ktree, e.sTree);
            //child function return -1 if sTree no have any child
            //no tiene hijo si en la posición e.sTreee hay un 0.

            for (int i = 0; i < (K * K); i++) {
                if (!esCero(rep->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = maxDistSinSqrt(q, child.area);
                    /*if (child.priority <= max) {
                        //return max;
                        //cout << "RETORNO 2" << endl;
                        return;
                    }*/
                    minDis2Child = minDistSinSqrt(q, child.area);
                    if (minDis2Child < c) {
                        pQueue.push(child);
                    }
                }
            }
        }
    }
    //hd = c;
    //cout << c << endl;
    return c;
}
//*********************************************************************************************
/*UTILIZADA PARA LA FUNCION ORIGINAL DE LA TESIS; QUE NO UTILIZA CMIN*/
void recursiveGetRankHD2(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
                         uint dp, uint dq, int x, int l, uint &n, double &cmax, Snapshot *B) {
    MREP2 *rep = snap->ktree;
    uint i = 0, j, leaf;
    uint y, p1new, p2new, q1new, q2new;
    unsigned long int divlevel;
    if (l == rep->maxLevel) {
        //recorrido por el bitarray leavesInf
        leaf = x + i * p1;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                leaf = x + j;
                if (rep->bl->access(leaf)) {
                    //dp + i => posición X del objeto econtrado
                    //dq + j => posición Y del objeto encontrado.
                    //snap->labels->getObjects(rep->bl->rank1(leaf), dp + i,dq + j, Oid, X, Y, n);
                    //cout << dp + i << endl;
                    //cout << dq + j << endl;
                    nnMax(B, Point(dp + i, dq + j), cmax);
                    //cout << cmax << endl;
                }
            }
            leaf += K;
        }

    }

    if ((l == rep->maxLevel - 1)
        && (rep->bn->access(x - rep->bt->getLength()))) {
        //recorrido por el bitarray bn
        y = (rep->bn->rank1(x - rep->bt->getLength()) - 1) * K * K;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                recursiveGetRankHD2(snap, 0, 0, 0, 0, dp + i, dq + j,
                                    y + K * i + j, l + 1, n, cmax, B);
            }
        }

    }
    if ((x == -1) || ((l < rep->maxLevel - 1) && (rep->bt->access(x)))) {
        //recorrido por el bitarray bt
        y = (x == -1) ? 0 : rep->bt->rank1(x) * K * K;
        divlevel = rep->div_level_table[l + 1];
        for (i = p1 / divlevel; i <= p2 / divlevel; i++) {
            p1new = 0;
            if (i == p1 / divlevel)
                p1new = p1 % divlevel;
            p2new = divlevel - 1;
            if (i == p2 / divlevel)
                p2new = p2 % divlevel;
            for (j = q1 / divlevel; j <= q2 / divlevel; j++) {
                q1new = 0;
                if (j == q1 / divlevel)
                    q1new = q1 % divlevel;
                q2new = divlevel - 1;
                if (j == q2 / divlevel)
                    q2new = q2 % divlevel;
                recursiveGetRankHD2(snap, p1new, p2new, q1new, q2new,
                                    dp + divlevel * i, dq + divlevel * j, y + K * i + j,
                                    l + 1, n, cmax, B);
            }
        }

    }
}
/*************************************************************************************/
/*UTILIZADA PARA LA FUNCION QUE UTILIZA CMIN o PNN*/
void recursiveGetRankHD2Opt(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
                            uint dp, uint dq, int x, int l, uint &n, double &cmax, Snapshot *B, Point &pnn,
                            double &cont) {
    MREP2 *rep = snap->ktree;
    uint i = 0, j, leaf;
    uint y, p1new, p2new, q1new, q2new;
    double d = 0.0;
    unsigned long int divlevel;
    if (l == rep->maxLevel) {
        //recorrido por el bitarray leavesInf
        leaf = x + i * p1;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                leaf = x + j;
                if (rep->bl->access(leaf)) {
                    Point p;
                    p.setX(dp + i);
                    p.setY(dq + j);
                    //d = (pnn.getX() - p.getX())*(pnn.getX() - p.getX()) + (pnn.getY() - p.getY())*(pnn.getY() - p.getY());
                    d = (pnn.getX() - p.getX()) * (pnn.getX() - p.getX()) +
                        (pnn.getY() - p.getY()) * (pnn.getY() - p.getY());
                    //cout << pnn.getX() << endl;
                    //cout << sqrt(d) << endl;
                    //cout << sqrt(cmax) << endl;
                    if (cmax < d) {
                        nnMaxOpt(B, p, cmax, pnn, d);
                        cont = cont + 1;
                    }
                }
            }
            leaf += K;
        }

    }

    if ((l == rep->maxLevel - 1)
        && (rep->bn->access(x - rep->bt->getLength()))) {
        //recorrido por el bitarray bn
        y = (rep->bn->rank1(x - rep->bt->getLength()) - 1) * K * K;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                recursiveGetRankHD2Opt(snap, 0, 0, 0, 0, dp + i, dq + j,
                                       y + K * i + j, l + 1, n, cmax, B, pnn, cont);
            }
        }

    }
    if ((x == -1) || ((l < rep->maxLevel - 1) && (rep->bt->access(x)))) {
        //recorrido por el bitarray bt
        y = (x == -1) ? 0 : rep->bt->rank1(x) * K * K;
        divlevel = rep->div_level_table[l + 1];
        for (i = p1 / divlevel; i <= p2 / divlevel; i++) {
            p1new = 0;
            if (i == p1 / divlevel)
                p1new = p1 % divlevel;
            p2new = divlevel - 1;
            if (i == p2 / divlevel)
                p2new = p2 % divlevel;
            for (j = q1 / divlevel; j <= q2 / divlevel; j++) {
                q1new = 0;
                if (j == q1 / divlevel)
                    q1new = q1 % divlevel;
                q2new = divlevel - 1;
                if (j == q2 / divlevel)
                    q2new = q2 % divlevel;
                recursiveGetRankHD2Opt(snap, p1new, p2new, q1new, q2new,
                                       dp + divlevel * i, dq + divlevel * j, y + K * i + j,
                                       l + 1, n, cmax, B, pnn, cont);
            }
        }

    }
}
/***********************************************************************************/
//arreglo donde la posición 0 tiene el total de elementos (n) y comienza en 1 y
//termina en n. n es un parámetro de salida.
/*void rangeQueryHD(Snapshot * snap, uint p1, uint p2, uint q1, uint q2,
uint * &Oid, uint * &X, uint * &Y, uint &n) {
	Oid = new uint[snap->totObj]();
	X = new uint[snap->totObj]();
	Y = new uint[snap->totObj]();
	n = 0;
	recursiveGetRankHD(snap, p1, p2, q1, q2, 0, 0, -1, -1, Oid, X, Y, n);
}*/
/**********************************************************/
/*FUNCION QUE UTILIZA PNN o CMIN*/
double hausdorffDistHDK2(Snapshot *C, Snapshot *D) {
    uint m = pow(K, C->ktree->maxLevel + 1) - 1;
    uint n;
    double superior = std::numeric_limits<int>::max();
    Point pnn;
    pnn.setX(superior);
    pnn.setY(superior);
    double cmax = 0.0;
    double cont;
    recursiveGetRankHD2Opt(C, 0, m, 0, m, 0, 0, -1, -1, n, cmax, D, pnn, cont);
    //cout << "Contador Optimizado: " << cont << endl;
    return sqrt(cmax);
}
/***************************************************************************/
/*double hausdorffDistOpt_(Snapshot * C, Snapshot * D) {
	uint m = pow(K, C->ktree->maxLevel+1) - 1;
	uint n;
	double superior = std::numeric_limits<int>::max();
	//uint * Oid;
	Point pnn;
	pnn.setX(superior);
	pnn.setY(superior);
	Point q;//Para obtener algun punto del primer conjunto
	//double cmax = 0.0;
	//Necesito Obtener un punto p del primer conjunto
	firstLeaf(C, 0, m, 0, m, 0, 0, -1, -1, n, q);
	double cmax = nnMaxSinCmax(D,q);
	cout << cmax << endl;
	double cont;
	recursiveGetRankHD2Opt(C, 0, m, 0, m, 0, 0, -1, -1, n, cmax, D, pnn, cont);
	cout << "Contador Optimizado: " << cont << endl;
	return sqrt(cmax);
}*/
//*********************************************************************************************
/**********************************************************/
/*FUNCION QUE UTILIZA NO UTILIZA CMIN, VERSION NORMAL DE LA TESIS*/
double hausdorffDist(Snapshot *A, Snapshot *B) {
    uint m = pow(K, A->ktree->maxLevel + 1) - 1;
    uint n;
    //uint * Oid;
    double cmax = 0.0;
    recursiveGetRankHD2(A, 0, m, 0, m, 0, 0, -1, -1, n, cmax, B);
    //delete m;
    return sqrt(cmax);
}

//*********************************************************************************************
//genera un número aleatorio entre x e y
int aleatorioEntreHd(int x, int y) {

    return x + ((rand() + 0.0f) / (RAND_MAX + 0.0f)) * (y - x);
    //return 1+rand()%(y-x);
}

//Randomiza un conjunto de Puntos
void randomization(Point Br[]) {
    //int largo = sizeof(A)/sizeof(*A);
    //random_shuffle(std::begin(Br), std::end(Br));
    random_shuffle(&Br[0], &Br[10]);
    /*int largo = 10;
    cout << "Largo" << largo << endl;
       //Point * out = new Point[largo];
    //out = A;
    int j;
    Point aux;
    //int largo = sizeof(out)/sizeof(*out)
    for (int i = 0; i < largo; i++) {
        j = (int) (aleatorioEntreHd(0, largo));
        cout << "Jota" << j << endl;
        aux = Br[i];
        Br[i] = Br[j];
        Br[j] = aux;
    }*/
    //return out;
}

void recursiveGetRankHD(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
                        uint dp, uint dq, int x, int l, uint *&Oid, uint &n, Point aux[], int &z) {
    MREP2 *rep = snap->ktree;
    uint i = 0, j, leaf;
    uint y, p1new, p2new, q1new, q2new;
    //uint z = 0;
    unsigned long int divlevel;
    if (l == rep->maxLevel) {
        //recorrido por el bitarray leavesInf
        leaf = x + i * p1;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                leaf = x + j;
                if (rep->bl->access(leaf)) {
                    //dp + i => posición X del objeto econtrado
                    //dq + j => posición Y del objeto encontrado.
                    //snap->labels->getObjects(rep->bl->rank1(leaf), dp + i, dq + j, Oid, X, Y, n);
                    aux[z] = Point(dp + i, dq + j);
                    z = z + 1;
                    //cout << z << endl;
                    //cout << dp + i << endl;
                    //cout << dq + j << endl;
                }
            }
            leaf += K;
        }

    }

    if ((l == rep->maxLevel - 1)
        && (rep->bn->access(x - rep->bt->getLength()))) {
        //recorrido por el bitarray bn
        y = (rep->bn->rank1(x - rep->bt->getLength()) - 1) * K * K;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                recursiveGetRankHD(snap, 0, 0, 0, 0, dp + i, dq + j,
                                   y + K * i + j, l + 1, Oid, n, aux, z);
            }
        }

    }
    if ((x == -1) || ((l < rep->maxLevel - 1) && (rep->bt->access(x)))) {
        //recorrido por el bitarray bt
        y = (x == -1) ? 0 : rep->bt->rank1(x) * K * K;
        divlevel = rep->div_level_table[l + 1];
        for (i = p1 / divlevel; i <= p2 / divlevel; i++) {
            p1new = 0;
            if (i == p1 / divlevel)
                p1new = p1 % divlevel;
            p2new = divlevel - 1;
            if (i == p2 / divlevel)
                p2new = p2 % divlevel;
            for (j = q1 / divlevel; j <= q2 / divlevel; j++) {
                q1new = 0;
                if (j == q1 / divlevel)
                    q1new = q1 % divlevel;
                q2new = divlevel - 1;
                if (j == q2 / divlevel)
                    q2new = q2 % divlevel;
                recursiveGetRankHD(snap, p1new, p2new, q1new, q2new,
                                   dp + divlevel * i, dq + divlevel * j, y + K * i + j,
                                   l + 1, Oid, n, aux, z);
            }
        }

    }
}

void recursiveGetRankHDOriginal(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
                                uint dp, uint dq, int x, int l, uint *&Oid, uint &n, uint *X, uint *Y) {
    MREP2 *rep = snap->ktree;
    uint i = 0, j, leaf;
    uint y, p1new, p2new, q1new, q2new;
    //uint z = 0;
    unsigned long int divlevel;
    if (l == rep->maxLevel) {
        //recorrido por el bitarray leavesInf
        leaf = x + i * p1;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                leaf = x + j;
                if (rep->bl->access(leaf)) {
                    //dp + i => posición X del objeto econtrado
                    //dq + j => posición Y del objeto encontrado.
                    snap->labels->getObjects(rep->bl->rank1(leaf), dp + i, dq + j, Oid, X, Y, n);
                    //aux[z] = Point(dp + i, dq + j);
                    //z = z + 1;
                    //cout << z << endl;
                    cout << dp + i << endl;
                    cout << dq + j << endl;
                }
            }
            leaf += K;
        }

    }

    if ((l == rep->maxLevel - 1)
        && (rep->bn->access(x - rep->bt->getLength()))) {
        //recorrido por el bitarray bn
        y = (rep->bn->rank1(x - rep->bt->getLength()) - 1) * K * K;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                recursiveGetRankHDOriginal(snap, 0, 0, 0, 0, dp + i, dq + j,
                                           y + K * i + j, l + 1, Oid, n, X, Y);
            }
        }

    }
    if ((x == -1) || ((l < rep->maxLevel - 1) && (rep->bt->access(x)))) {
        //recorrido por el bitarray bt
        y = (x == -1) ? 0 : rep->bt->rank1(x) * K * K;
        divlevel = rep->div_level_table[l + 1];
        for (i = p1 / divlevel; i <= p2 / divlevel; i++) {
            p1new = 0;
            if (i == p1 / divlevel)
                p1new = p1 % divlevel;
            p2new = divlevel - 1;
            if (i == p2 / divlevel)
                p2new = p2 % divlevel;
            for (j = q1 / divlevel; j <= q2 / divlevel; j++) {
                q1new = 0;
                if (j == q1 / divlevel)
                    q1new = q1 % divlevel;
                q2new = divlevel - 1;
                if (j == q2 / divlevel)
                    q2new = q2 % divlevel;
                recursiveGetRankHDOriginal(snap, 0, 0, 0, 0, dp + i, dq + j,
                                           y + K * i + j, l + 1, Oid, n, X, Y);
            }
        }

    }
}

void rangeQueryHD(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
                  uint *&Oid, uint &n, Point aux[]) {
    Oid = new uint[snap->totObj]();
    //X = new uint[snap->totObj]();
    //Y = new uint[snap->totObj]();
    n = 0;
    int z = 0;
    recursiveGetRankHD(snap, p1, p2, q1, q2, 0, 0, -1, -1, Oid, n, aux, z);
}

void rangeQueryHDOriginal(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
                          uint *&Oid, uint *&X, uint *&Y, uint &n) {
    Oid = new uint[snap->totObj]();
    X = new uint[snap->totObj]();
    Y = new uint[snap->totObj]();
    n = 0;
    recursiveGetRank(snap, p1, p2, q1, q2, 0, 0, -1, -1, Oid, X, Y, n);
}
void rangeQueryHDOriginal2(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
                          uint *&Oid, uint *&X, uint *&Y, uint &n) {
    Oid = new uint[snap->totObj]();
    X = new uint[snap->totObj]();
    Y = new uint[snap->totObj]();
    n = 0;
    recursiveGetRank2(snap, p1, p2, q1, q2, 0, 0, -1, -1, Oid, X, Y, n);
}
/****************************************************************************/
/*FUNCION RANDOM QUE SE UTILIZO SOLO DE PRUEBA*/
/*double hausdorffDistRandom(Snapshot * E, Snapshot * F) {
//obtenemos todos los puntos de A.
   uint * Oid;
   //uint * X;
   //uint * Y;
   uint m = pow(K, E->ktree->maxLevel+1) - 1;
   uint n;
   Point * aux = new Point[E->totObj];
   int contran = 0;
   //cout << m << " fsdf " << n << endl;
   //rangeQueryHD(E, 0, m, 0, m, Oid, X, Y, n, aux); //No obtiene todos los puntos
   rangeQueryHD(E, 0, m, 0, m, Oid, n, aux);
   Point * Br = aux;
   randomization(Br);//randomizar B
   double cmax = 0.0;
   double d = 0.0;
   double superior = std::numeric_limits<int>::max();
   Point pnn;
   pnn.setX(superior);
   pnn.setY(superior);
   //pnn.setX(0);
   //pnn.setY(0);
   for (uint i = 0; i < E->totObj-1; i++) {
       //nnMax(B, Point(X[i], Y[i]), cmax);

       //d = sqrt((pnn.getX() - Br[i].getX())*(pnn.getX() - Br[i].getX()) + (pnn.getY() - Br[i].getY())*(pnn.getY() - Br[i].getY()));
       d = (pnn.getX() - Br[i].getX())*(pnn.getX() - Br[i].getX()) + (pnn.getY() - Br[i].getY())*(pnn.getY() - Br[i].getY());
       if(cmax < d){
           nnMaxOpt(F, Br[i], cmax, pnn, d);
           contran = contran + 1;
           //cout << "Elementos Originales: " << aux[i].getX() << endl;
           //cout << "Elementos Random: " << Br[i].getX() << endl;
       }
   }
   delete[] Oid;
   delete[] Br;
   //delete[] aux;
   //delete[] X;
   //delete[] Y;
   cout << "Contador Random: " << contran << endl;
   return sqrt(cmax);

}*/
//*********************************************************************************************
/*PARA PROBAR CUANDO NO UTILIZAMOS PODAS*/
void recursiveGetRankHDSinCmax(Snapshot *snap, uint p1, uint p2, uint q1, uint q2,
                               uint dp, uint dq, int x, int l, uint &n, double &hd, Snapshot *B) {
    MREP2 *rep = snap->ktree;
    uint i = 0, j, leaf;
    uint y, p1new, p2new, q1new, q2new;
    double cmin;
    unsigned long int divlevel;
    if (l == rep->maxLevel) {
        //recorrido por el bitarray leavesInf
        leaf = x + i * p1;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                leaf = x + j;
                if (rep->bl->access(leaf)) {
                    //dp + i => posición X del objeto econtrado
                    //dq + j => posición Y del objeto encontrado.
                    //snap->labels->getObjects(rep->bl->rank1(leaf), dp + i,dq + j, Oid, X, Y, n);
                    //cout << dp + i << endl;
                    //cout << dq + j << endl;

                    cmin = nnMaxSinCmax(B, Point(dp + i, dq + j));
                    if (cmin > hd) hd = cmin;
                    //cout << hd << endl;
                }
            }
            leaf += K;
        }

    }

    if ((l == rep->maxLevel - 1)
        && (rep->bn->access(x - rep->bt->getLength()))) {
        //recorrido por el bitarray bn
        y = (rep->bn->rank1(x - rep->bt->getLength()) - 1) * K * K;
        for (i = p1; i <= p2; i++) {
            for (j = q1; j <= q2; j++) {
                recursiveGetRankHDSinCmax(snap, 0, 0, 0, 0, dp + i, dq + j,
                                          y + K * i + j, l + 1, n, hd, B);
            }
        }

    }
    if ((x == -1) || ((l < rep->maxLevel - 1) && (rep->bt->access(x)))) {
        //recorrido por el bitarray bt
        y = (x == -1) ? 0 : rep->bt->rank1(x) * K * K;
        divlevel = rep->div_level_table[l + 1];
        for (i = p1 / divlevel; i <= p2 / divlevel; i++) {
            p1new = 0;
            if (i == p1 / divlevel)
                p1new = p1 % divlevel;
            p2new = divlevel - 1;
            if (i == p2 / divlevel)
                p2new = p2 % divlevel;
            for (j = q1 / divlevel; j <= q2 / divlevel; j++) {
                q1new = 0;
                if (j == q1 / divlevel)
                    q1new = q1 % divlevel;
                q2new = divlevel - 1;
                if (j == q2 / divlevel)
                    q2new = q2 % divlevel;
                recursiveGetRankHDSinCmax(snap, p1new, p2new, q1new, q2new,
                                          dp + divlevel * i, dq + divlevel * j, y + K * i + j,
                                          l + 1, n, hd, B);
            }
        }

    }
}

/*FUNCION QUE NO USA PODAS*/
double hausdorffDistSinCmax(Snapshot *G, Snapshot *H) {
    uint m = pow(K, G->ktree->maxLevel + 1) - 1;
    uint n;
    //uint * Oid;
    double hd = 0.0;
    recursiveGetRankHDSinCmax(G, 0, m, 0, m, 0, 0, -1, -1, n, hd, H);
    //delete m;
    return sqrt(hd);
}
//*********************************************************************************************
//*********************************************************************************************
//RESPALDO FUNCION ORIGINAL
/*double hausdorffDist_(Snapshot * A, Snapshot * B) {
//obtenemos todos los puntos de A.

	uint * Oid;
	uint * X;
	uint * Y;
	uint m = pow(K, A->ktree->maxLevel+1) - 1;
	uint n;
	//cout << m << " fsdf " << n << endl;
	rangeQueryHD(A, 0, m, 0, m, Oid, X, Y, n); //No obtiene todos los puntos

	double cmax = 0.0;
	for (uint i = 0; i < n; i++) {
		nnMax(B, Point(X[i], Y[i]), cmax);
		//cout << "Elementos: " << n << endl;
	}
	delete[] Oid;
	delete[] X;
	delete[] Y;
	return cmax;

}*/
//*********************************************************************************************
//VERSION FINAL
/*void hausdorffDistVersionNueva(Snapshot * A, Snapshot * B) {
//obtenemos todos los puntos de A.
	unsigned tr0, tr1;
	//Aqui Colocamos la medicion de Tiempo
	tr0=clock();//DEBEMOS PEDIR A MIGUEL PARA OBTENER UN PUNTO DE INMEDIATO
	uint * Oid;
	uint * X;
	uint * Y;
	uint m = pow(K, A->ktree->maxLevel+1) - 1;
	uint n;
	/********************/
//MOMENTAMEOS MIENTRAS PREGUNTAMOS A MIGUEL
/*Point pnn;
double superior = std::numeric_limits<int>::max();
pnn.setX(superior);
pnn.setY(superior);
uint * Oid1;
uint * X1;
uint * Y1;
uint m1 = pow(K, B->ktree->maxLevel+1) - 1;
uint n1;
rangeQueryHDOriginal(B, 0, m1, 0, m1, Oid1, X1, Y1, n1);
pnn.setX(X1[5]);
pnn.setY(Y1[5]);



//rangeQueryHDOriginal(A, 0, m, 0, m, Oid, X, Y, n); //No obtiene todos los puntos
rangeQueryHDOriginal(A, 0, m, 0, m, Oid, X, Y, n);
//rangeQueryHDOriginal(B, 0, m, 0, m, Oid, X1, Y1, n);
double cmax = 0.0 , d = 0.0;
for (uint i = 0; i < n; i++) {
    Point p;
    p.setX(X[i]);
    p.setY(Y[i]);
    d = (pnn.getX() - p.getX())*(pnn.getX() - p.getX()) + (pnn.getY() - p.getY())*(pnn.getY() - p.getY());
    if(cmax < d){
        nnMaxOpt(B, p, cmax, pnn, d);

    }
}
//Fin Medicion
cout << "Distancia de Hausdorff Poda en un K2-Tree: : " << sqrt(cmax) << endl;
tr1 = clock();
double timeran = (double(tr1-tr0)*1000/CLOCKS_PER_SEC);
cout << "Execution Time: " << timeran << endl;
cout << "--------------------------------------------------------------------" << endl;
delete[] Oid;
delete[] X;
delete[] Y;
//return 0;

}*/
//*********************************************************************************************
/*void isCandidateTest(ElementQueue eq, Snapshot * B,  double &supermax){
	priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
	PriorityQueue pq;
	long int subTreeIndex;
	ElementQueue e;
	//ElementQueue child;
	e.sTree = -1;
	e.area.x1 = 0;
	//el 2 es porque k = 2, reemplazar por K.
	e.area.x2 = pow(K, B->ktree->maxLevel+1) - 1;
	e.area.y1 = e.area.x1;
	e.area.y2 = e.area.x2;
	pQueue.push(e);
	while (!pQueue.empty()) {
			e = pQueue.top();
			pQueue.pop();
			if (isLeaf(B->ktree, e.sTree)) {
				//return true;
				//return e;// Si es hoja, devolvemos el cuadrante, que en este caso es el que tiene la menor distancia.
				//Alli debe estar el vecino mas cercano, del otro cuadrante
				//return pQueue;
				pq.pPila = pQueue;
				pq.bandera = 1;
				//return pq;
				return;

			}
			else{
				subTreeIndex = firstChild(B->ktree, e.sTree);
				for (int i = 0; i < (K * K); i++) {
					if (!esCero(B->ktree, subTreeIndex + i)) {
						ElementQueue child;
						child.sTree = subTreeIndex + i;
						child.area = getSubArea(e.area, i);
						//child.priority = maxDistSinSqrt(q, child.area);
						child.priority = maxDistAreas(eq.area, child.area);
						//cout << "Priority " << sqrt(child.priority) << endl;
						//cout << "Supermax " << sqrt(supermax) << endl;
						if (child.priority <= supermax) {
							//return max;
							//rpq.banderaeturn false;
							pq.bandera = -1; //No es candidato
							//return pq;
							return;
							//return pQueue;
						}
						pQueue.push(child);

					}
				}
			}
	}
}*/
//*********************************************************************************************
double isCandidate(ElementQueue eq, Snapshot *B, double &supermax) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    long int subTreeIndex;
    ElementQueue e;
    e.sTree = -1;
    e.area.x1 = 0;
    //el 2 es porque k = 2, reemplazar por K.
    e.area.x2 = pow(K, B->ktree->maxLevel + 1) - 1;
    e.area.y1 = e.area.x1;
    e.area.y2 = e.area.x2;
    pQueue.push(e);
    while (!pQueue.empty()) {
        e = pQueue.top();
        pQueue.pop();
        if (isLeaf(B->ktree, e.sTree)) {
            //return true;
            //return -1;
            return e.priority;
        } else {
            subTreeIndex = firstChild(B->ktree, e.sTree);
            for (int i = 0; i < (K * K); i++) {
                if (!esCero(B->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = maxDistAreas(eq.area, child.area);
                    if (child.priority <= supermax) {
                        //return child.priority;
                        //return false;
                        return -1;
                    }
                    pQueue.push(child);
                }
            }
        }
    }
}
/*
void isCandidateLeaf(ElementQueue eq, Snapshot * B,  double &supermax, PriorityQueue &pq){
	//DEBE RETORNAR UNA ESTRUCTURA CON LA COLA Y ELEMENTQUEUE DE ESA COLA
	priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
	//PriorityQueue pq;
	long int subTreeIndex;
	ElementQueue e;
	//ElementQueue child;
	e.sTree = -1;
	e.area.x1 = 0;
	//el 2 es porque k = 2, reemplazar por K.
	e.area.x2 = pow(K, B->ktree->maxLevel+1) - 1;
	e.area.y1 = e.area.x1;
	e.area.y2 = e.area.x2;
	pQueue.push(e);
	while (!pQueue.empty()) {
			e = pQueue.top();
			pQueue.pop();
			if (isLeaf(B->ktree, e.sTree)) {
				//return true;
				//return e;// Si es hoja, devolvemos el cuadrante, que en este caso es el que tiene la menor distancia.
				//Alli debe estar el vecino mas cercano, del otro cuadrante
				//return pQueue;
				pq.pPila = pQueue;
				pq.bandera = 1;
				//return pq;
				return;

			}
			else{
				subTreeIndex = firstChild(B->ktree, e.sTree);
				for (int i = 0; i < (K * K); i++) {
					if (!esCero(B->ktree, subTreeIndex + i)) {
						ElementQueue child;
						child.sTree = subTreeIndex + i;
						child.area = getSubArea(e.area, i);
						//child.priority = maxDistSinSqrt(q, child.area);
						child.priority = maxDistAreas(eq.area, child.area);
						//cout << "Priority " << sqrt(child.priority) << endl;
						//cout << "Supermax " << sqrt(supermax) << endl;
						if (child.priority <= supermax) {
							//return max;
							//rpq.banderaeturn false;
							pq.bandera = -1; //No es candidato
							//return pq;
							return;
							//return pQueue;
						}
						pQueue.push(child);

					}
				}
			}
	}
}*/
//FUNCION QUE PODA EN AMBOS K2-TREE
/*void hdk( Snapshot * A, Snapshot * B,  double &supermax){
	priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
	priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> auxQueue;
	//queue<ElementQueue, std::vector<ElementQueue>> pQueue;
	//Calculamos la region de A
	ElementQueue aa;
	//CandidateElement aa, ba;
	aa.sTree = -1;
	aa.area.x1 = 0;
	//el 2 es porque k = 2, reemplazar por K.
	aa.area.x2 = pow(K, A->ktree->maxLevel+1) - 1;
	aa.area.y1 = aa.area.x1;
    aa.area.y2 = aa.area.x2;

    long int subTreeIndex;
    PriorityQueue  pfilaB;
    //int cont = 0;
    //Aqui tenemos que ver la forma como controlar que revise todo las regiones
    pQueue.push(aa);
    while (!pQueue.empty()) {
		aa = pQueue.top();
		pQueue.pop();
		if (isLeaf(A->ktree, aa.sTree)) {
				//si es hoja de último nivel
				cout << "Paso1" << endl;
				Point q;
				q.setX(aa.area.x1);
				q.setY(aa.area.y1);
				double aux = supermax;
				/*auxQueue = pfilaB.pPila;
				ElementQueue aux1 = auxQueue.top();
				cout << sqrt(aux1.priority) << endl;*/
/*nnMaxQueue(pfilaB, B, q, supermax);
PriorityQueue  pfilaB;
if ((aux >= supermax)) { //supermax es pasada por referencia, se actualiza sola
    supermax = aux;//Ahora bien, si no mejoramos supermax, debemos volverla a su valor anterior
}
}else{
subTreeIndex = firstChild(A->ktree, aa.sTree);
for(int i = 0; i < (K * K); i++){
if(!esCero(A->ktree, subTreeIndex + i)){
        ElementQueue child;
        child.sTree = subTreeIndex + i;
        child.area = getSubArea(aa.area,i);

        //cout << "Paso2" << endl;
        //SI NO ES HIJO REHACEMOS LA COLA
        if (isLeaf(A->ktree, aa.sTree)) {
            //Si es una hoja, entonces
            PriorityQueue  pfilaB;
            isCandidateLeaf(child,B,supermax, pfilaB);
        }
        else{
            isCandidate(child,B,supermax);
        }
        /*auxQueue = pfilaB.pPila;
        ElementQueue aux1 = auxQueue.top();
        cout << sqrt(aux1.priority) << endl;*/
/*if(pfilaB.bandera != -1 ){ // SI es -1 es que no encontro ninguno
        pQueue.push(child);

}
}
}
}
}


}*/

void  saveMaxPqueue(unsigned long num) {
    if (num > heapMaxElements){
        heapMaxElements = num;
    }
}
void  saveExitHeap(unsigned long num) {
   exitHeapElements = num;
}
//FUNCION QUE PODA EN AMBOS K2-TREE USANDO UNA MAXHEAP


void hdkMaxHeap(Snapshot *A, Snapshot *B, double &supermax) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, maxHeapComparatorHD> pQueue;
    //Calculamos la region de A
    ElementQueue aa;
    aa.sTree = -1;
    aa.area.x1 = 0;
    //el 2 es porque k = 2, reemplazar por K.
    aa.area.x2 = pow(K, A->ktree->maxLevel + 1) - 1;
    aa.area.y1 = aa.area.x1;
    aa.area.y2 = aa.area.x2;
   // aa.priority = A->ktree->numberOfNodes * sqrt(2); // la diagonal es la mayor distancia posible..
    int contador = 0;
    long int subTreeIndex;
    PriorityQueue pfilaB;
    cout <<"insertamos en pqueue: ";
    printElementQueue(aa);
    
    pQueue.push(aa);
    while (!pQueue.empty()) {
        saveMaxPqueue(pQueue.size()); // para saber el maximo ocupado por pqueue y reportar el espacio ocupado..
        aa = pQueue.top();
        cout <<"pqueue top"<< endl;
        pQueue.pop();
        //Aqui verificamos si es hoja.
        if (isLeaf(A->ktree, aa.sTree)) {
            cout << "Hoja" << endl;
            Point q;
            q.setX(aa.area.x1);
            q.setY(aa.area.y1);
            std::cout << "Point: (" << aa.area.x1 << ", " << aa.area.y1 << ")\n";
            double aux = supermax;
            cout <<"primer supermax: " << supermax << endl;
            contador++;
            cout << "entrando a nnMax" << endl;
            nnMax(B, q, supermax);
            cout <<"segundo supermax: " << supermax << endl;
            if ((aux >= supermax)) { //supermax es pasada por referencia, se actualiza sola
                cout <<"hizo el cambio de supermax" << endl;
                supermax = aux;//Ahora bien, si no mejoramos supermax, debemos volverla a su valor anterior
            }
        } else {
            cout << "No es hoja" << endl;
            subTreeIndex = firstChild(A->ktree, aa.sTree);
            for (int i = 0; i < (K * K); i++) {
                if (!esCero(A->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(aa.area, i);
                    if ((child.priority = isCandidate(child, B, supermax)) != -1) {
                        cout <<"insertamos en pqueue: ";
                        printElementQueue(child);
                        pQueue.push(child);
                    }
                }
            }
        }
    }
    saveExitHeap(pQueue.size()); //agregado por fernando.
    //cout << "N° de Puntos MaxHeap: " << contador << endl;
}
void hdkMaxHeapv2(Snapshot *A, Snapshot *B, double &supermax) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, maxHeapComparatorHD> pQueue;
    //Calculamos la region de A
    ElementQueue aa;
    aa.sTree = -1;
    aa.area.x1 = 0;
    //el 2 es porque k = 2, reemplazar por K.
    aa.area.x2 = pow(K, A->ktree->maxLevel + 1) - 1;
    aa.area.y1 = aa.area.x1;
    aa.area.y2 = aa.area.x2;
    aa.priority = A->ktree->numberOfNodes * sqrt(2); // la diagonal es la mayor distancia posible..
    int contador = 0;
    long int subTreeIndex;
    PriorityQueue pfilaB;
    pQueue.push(aa);
    while (!pQueue.empty()) {
        aa = pQueue.top();
        pQueue.pop();
        /* Si la mayor distancia posible del cuadrante aa es menor que supermax
         * ningun cuadrante del heap (pqueue) mejorará el resultado conocido y por tanto se acaba la busqueda.
          * */
            if(aa.priority <= supermax){
                return;
        }
        //Aqui verificamos si es hoja.
        if (isLeaf(A->ktree, aa.sTree)) {
            Point q;
            q.setX(aa.area.x1);
            q.setY(aa.area.y1);
            double aux = supermax;
            nnMax(B, q, supermax);
            if ((aux >= supermax)) { //supermax es pasada por referencia, se actualiza sola
                supermax = aux;//Ahora bien, si no mejoramos supermax, debemos volverla a su valor anterior
            }
        } else {
            subTreeIndex = firstChild(A->ktree, aa.sTree);
            for (int i = 0; i < (K * K); i++) {
                if (!esCero(A->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(aa.area, i);
                    if ((child.priority = isCandidate(child, B, supermax)) != -1) {
                        pQueue.push(child);
                    }
                }
            }
        }
    }

}

//FUNCION QUE PODA EN AMBOS K2-TREE USANDO UNA MAXHEAP
void hdkMinHeap(Snapshot *A, Snapshot *B, double &supermax) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    //Calculamos la region de A
    ElementQueue aa;
    aa.sTree = -1;
    aa.area.x1 = 0;
    //el 2 es porque k = 2, reemplazar por K.
    aa.area.x2 = pow(K, A->ktree->maxLevel + 1) - 1;
    aa.area.y1 = aa.area.x1;
    aa.area.y2 = aa.area.x2;
    int contador = 0;
    long int subTreeIndex;
    PriorityQueue pfilaB;
    pQueue.push(aa);
    while (!pQueue.empty()) {
        aa = pQueue.top();
        pQueue.pop();
        //Aqui verificamos si es hoja.
        if (isLeaf(A->ktree, aa.sTree)) {
            //cout << "Hoja" << endl;
            Point q;
            q.setX(aa.area.x1);
            q.setY(aa.area.y1);
            double aux = supermax;
            contador++;
            nnMax(B, q, supermax);
            if ((aux >= supermax)) { //supermax es pasada por referencia, se actualiza sola
                supermax = aux;//Ahora bien, si no mejoramos supermax, debemos volverla a su valor anterior
            }
        } else {
            subTreeIndex = firstChild(A->ktree, aa.sTree);
            for (int i = 0; i < (K * K); i++) {
                if (!esCero(A->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(aa.area, i);
                    if ((child.priority = isCandidate(child, B, supermax)) != -1) {
                        pQueue.push(child);

                    }

                }
            }
        }

    }
  //  cout << "N° de Puntos MIN HEAP: " << contador << endl;

}

//FUNCION QUE PODA EN AMBOS K2-TREE USANDO UNA FILA
void hdkPila(Snapshot *A, Snapshot *B, double &supermax) {
    //priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    std::stack<ElementQueue> pQueue;
    //Calculamos la region de A
    ElementQueue aa;
    aa.sTree = -1;
    aa.area.x1 = 0;
    //el 2 es porque k = 2, reemplazar por K.
    aa.area.x2 = pow(K, A->ktree->maxLevel + 1) - 1;
    aa.area.y1 = aa.area.x1;
    aa.area.y2 = aa.area.x2;
    long int subTreeIndex;
    int contador = 0;
    pQueue.push(aa);
    while (!pQueue.empty()) {
        aa = pQueue.top();
        pQueue.pop();
        //Aqui verificamos si es hoja.
        if (isLeaf(A->ktree, aa.sTree)) {
            //cout << "Hoja" << endl;
            Point q;
            q.setX(aa.area.x1);
            q.setY(aa.area.y1);
            double aux = supermax;
            contador++;
            nnMax(B, q, supermax);
            if ((aux >= supermax)) { //supermax es pasada por referencia, se actualiza sola
                supermax = aux;//Ahora bien, si no mejoramos supermax, debemos volverla a su valor anterior
            }
        } else {
            subTreeIndex = firstChild(A->ktree, aa.sTree);
            for (int i = 0; i < (K * K); i++) {
                if (!esCero(A->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(aa.area, i);
                    if ((child.priority = isCandidate(child, B, supermax)) != -1) {
                        pQueue.push(child);

                    }

                }
            }
        }

    }
  //  cout << "N° de Puntos Pila: " << contador << endl;

}

//FUNCION QUE PODA EN AMBOS K2-TREE USANDO UNA FILA
void hdkFila(Snapshot *A, Snapshot *B, double &supermax) {
    //priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    std::queue<ElementQueue> pQueue;
    //Calculamos la region de A
    ElementQueue aa;
    aa.sTree = -1;
    aa.area.x1 = 0;
    //el 2 es porque k = 2, reemplazar por K.
    aa.area.x2 = pow(K, A->ktree->maxLevel + 1) - 1;
    aa.area.y1 = aa.area.x1;
    aa.area.y2 = aa.area.x2;
    int contador = 0;
    long int subTreeIndex;
    pQueue.push(aa);
    while (!pQueue.empty()) {
        aa = pQueue.front();
        pQueue.pop();
        //Aqui verificamos si es hoja.
        if (isLeaf(A->ktree, aa.sTree)) {
            //cout << "Hoja" << endl;
            Point q;
            q.setX(aa.area.x1);
            q.setY(aa.area.y1);
            double aux = supermax;
            contador++;
            nnMax(B, q, supermax);
            if ((aux >= supermax)) { //supermax es pasada por referencia, se actualiza sola
                supermax = aux;//Ahora bien, si no mejoramos supermax, debemos volverla a su valor anterior
            }
        } else {
            subTreeIndex = firstChild(A->ktree, aa.sTree);
            for (int i = 0; i < (K * K); i++) {
                if (!esCero(A->ktree, subTreeIndex + i)) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(aa.area, i);


                    if ((child.priority = isCandidate(child, B, supermax)) != -1) {
                        pQueue.push(child);

                    }

                }
            }
        }

    }
  //  cout << "N° de Puntos Fila: " << contador << endl;

}
//FUNCION QUE PODA EN AMBOS K2-TREE
/*void hdk_( Snapshot * A, Snapshot * B,  double &supermax){
	priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
	priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> auxQueue;
	//Calculamos la region de A
	ElementQueue aa;
	//CandidateElement aa, ba;
	aa.sTree = -1;
	aa.area.x1 = 0;
	//el 2 es porque k = 2, reemplazar por K.
	aa.area.x2 = pow(K, A->ktree->maxLevel+1) - 1;
	aa.area.y1 = aa.area.x1;
    aa.area.y2 = aa.area.x2;

    long int subTreeIndex;
    PriorityQueue  pfilaB;
    pQueue.push(aa);
    while (!pQueue.empty()) {
		aa = pQueue.top();
		pQueue.pop();
		subTreeIndex = firstChild(A->ktree, aa.sTree);
		for(int i = 0; i < (K * K); i++){
			if(!esCero(A->ktree, subTreeIndex + i)){
				ElementQueue child;
				child.sTree = subTreeIndex + i;
				child.area = getSubArea(aa.area,i);
				//Aqui verificamos si es hoja.
				if (isLeaf(A->ktree, aa.sTree)) {
							//si es hoja de último nivel
							Point q;
							q.setX(aa.area.x1);
							q.setY(aa.area.y1);
							double aux = supermax;
							nnMax( B, q, supermax);
							if ((aux >= supermax)) { //supermax es pasada por referencia, se actualiza sola
								supermax = aux;//Ahora bien, si no mejoramos supermax, debemos volverla a su valor anterior
							}
				}
				else{
					isCandidate(child,B,supermax, pfilaB);
					if(pfilaB.bandera != -1 ){ // SI es -1 es que no encontro ninguno
						pQueue.push(child);

					}
				}
			}
		}
    }

}
 */
void hausdorffDistHDK3(Snapshot *A, Snapshot *B) {

    double supermax = 0.0;
    unsigned tmaxheap0, tmaxheap1, tminheap0, tminheap1, tpila0, tpila1, tfila0, tfila1;
    //COMIENZA MEDICION
    tmaxheap0 = clock();
    /*Point *p = getObjectPos(A, 1);
    cout << p->getX() << endl;
    cout << p->getY() << endl;*/
    //nnMax(B,*p, supermax);
    //Aqui llamamos a la nueva funcion que poda en los 2 K2-Tree con MAXHEAP
    hdkMaxHeap(A, B, supermax);
    cout << "Distancia de Hausdorff Poda en los 2 K2-Tree MAXHEAP: " << sqrt(supermax) << endl;
    //FIN MEDICION
    tmaxheap1 = clock();
    double timeran = (double(tmaxheap1 - tmaxheap0) * 1000 / CLOCKS_PER_SEC);
    cout << "Execution Time: " << timeran << endl;
    cout << "--------------------------------------------------------------------" << endl;
    /***************************************************************************************/
    //Aqui llamamos a la nueva funcion que poda en los 2 K2-Tree con MINHEAP
    supermax = 0.0;
    //COMIENZA MEDICION
    tminheap0 = clock();
    //nnMax(B,*p, supermax);
    //Aqui llamamos a la nueva funcion que poda en los 2 K2-Tree con MINHEAP
    hdkMinHeap(A, B, supermax);
    cout << "Distancia de Hausdorff Poda en los 2 K2-Tree MINHEAP: " << sqrt(supermax) << endl;
    //FIN MEDICION
    tminheap1 = clock();
    timeran = (double(tminheap1 - tminheap0) * 1000 / CLOCKS_PER_SEC);
    cout << "Execution Time: " << timeran << endl;
    cout << "--------------------------------------------------------------------" << endl;
    /***************************************************************************************/
    //Aqui llamamos a la nueva funcion que poda en los 2 K2-Tree con una PILA
    supermax = 0.0;
    //COMIENZA MEDICION
    tpila0 = clock();
    //nnMax(B,*p, supermax);
    //Aqui llamamos a la nueva funcion que poda en los 2 K2-Tree con PILA
    hdkPila(A, B, supermax);
    cout << "Distancia de Hausdorff Poda en los 2 K2-Tree PILA: " << sqrt(supermax) << endl;
    //FIN MEDICION
    tpila1 = clock();
    timeran = (double(tpila1 - tpila0) * 1000 / CLOCKS_PER_SEC);
    cout << "Execution Time: " << timeran << endl;
    cout << "--------------------------------------------------------------------" << endl;
    /***************************************************************************************/
    //Aqui llamamos a la nueva funcion que poda en los 2 K2-Tree con una FILA
    supermax = 0.0;
    //COMIENZA MEDICION
    tfila0 = clock();
    //nnMax(B,*p, supermax);
    //Aqui llamamos a la nueva funcion que poda en los 2 K2-Tree con FILA
    hdkFila(A, B, supermax);
    cout << "Distancia de Hausdorff Poda en los 2 K2-Tree FILA: " << sqrt(supermax) << endl;
    //FIN MEDICION
    tfila1 = clock();
    timeran = (double(tfila1 - tfila0) * 1000 / CLOCKS_PER_SEC);
    cout << "Execution Time: " << timeran << endl;
    cout << "--------------------------------------------------------------------" << endl;
    /***************************************************************************************/

}

/*AGREGADO POR FERNANDO SANTOLAYA PARA HACER EXPERIMENTOS NUEVOS*/

double hausdorffDistHDK3MaxHeap(Snapshot *A, Snapshot *B) {
    double supermax = 0.0;
    hdkMaxHeap(A, B, supermax);
    return sqrt(supermax);
}
//Esta operación es la distancia Directa de hausdorff, es decir, d(A,B)
//agregado por Miguel Romero
double hausdorffDistHDK3MaxHeapv2(Snapshot *A, Snapshot *B) {
    double supermax = 0.0;
    hdkMaxHeapv2(A, B, supermax);
    return sqrt(supermax);
}

//agregado por Miguel Romero
//Distancia simétrica de Hausdorff, para compara con el algoritmo de Ryu-Kamata
double SymHDK2tree(Snapshot *A, Snapshot *B) {
    double supermax = 0.0;
    hdkMaxHeapv2(A, B, supermax);
    hdkMaxHeapv2(B, A, supermax);
    return sqrt(supermax);
}

double hausdorffDistHDK3MinHeap(Snapshot *A, Snapshot *B) {
    double supermax = 0.0;
    hdkMinHeap(A, B, supermax);
    return sqrt(supermax);
}

double hausdorffDistHDK3Pila(Snapshot *A, Snapshot *B) {
    double supermax = 0.0;
    hdkPila(A, B, supermax);
    return sqrt(supermax);
}

double hausdorffDistHDK3Fila(Snapshot *A, Snapshot *B) {
    double supermax = 0.0;
    hdkFila(A, B, supermax);
    return sqrt(supermax);
}

//*******************************************************************************************
//FUNCIONES DE HAUSDORFF DISTANCE SOLO PARA HACER EXPERIMENTOS
double abdelHDD(Point Er[], Point Br[], int largo) {
    double cmax, cmin, d;
    cmax = 0.0;
    //falta generar E=A\(A intersect B)
    //Point * Er = A;
    //Point * Br = B;
    //Er = A;
    //Br = B;
    //randomization(Er);//randomizar E
    //randomization(Br);//randomizar B
   // int largoEr = sizeof(Er) / sizeof(*Er);
   // int largoBr = sizeof(Br) / sizeof(*Br);

    for (int i = 0; i < largo; i++) {
        cmin = std::numeric_limits<double>::max();
        for (int j = 0; j < largo; j++) {
            d = (Er[i].getX() - Br[j].getX()) * (Er[i].getX() - Br[j].getX()) +
                (Er[i].getY() - Br[j].getY()) * (Er[i].getY() - Br[j].getY());
            //esta línea originalmente estaba después de la 48, pero
            //produce un error explicado más abajo.

            if (d < cmin) cmin = d;
            //originalmente es <, pero es más correcto <=
            if (d <= cmax) break;//early break!

        }
        //al hacer el "Early Break" hay que tener cuidado
        // con el valor que queda cmin, porque si queda en
        // infinito, entonces cmin, será mayor que cmax
        // artificialmente.  Esto ocurre en el algoritmo original
        // al hacer el break, justo con la primera distancia
        // una solución elegante, es evaluar primero (d < cmin),
        //de modo que cmin no termine en infinito sino en un valor
        //menor que cmax antes del break, salvandonos del error.
        if (cmin > cmax) cmax = cmin;
        //cout << Er[i].getX() << "," << Er[i].getY() << endl;
    }
    delete[] Er;
    delete[] Br;
    return sqrt(cmax);
}
std::vector<Point> extractPointK2tree(Snapshot *J, int nelementos) {
    uint *Oid1;
    uint *X1;
    uint *Y1;
    uint m1 = pow(K, J->ktree->maxLevel + 1) - 1;
    uint n1;
    // Llama a la función rangeQueryHDOriginal para obtener los puntos
    rangeQueryHDOriginal2(J, 0, m1, 0, m1, Oid1, X1, Y1, n1);
    std::vector<Point> Ar(n1);
    // Llena el vector con los puntos obtenidos
    for (uint i = 0; i < n1; i++) {
        Point p1;
        p1.setX(X1[i]);
        p1.setY(Y1[i]);
        Ar[i] = p1;
    }
    // Retorna el vector de puntos
    return Ar;
}

Point * extractPointK2tree2(Snapshot *J, int nelementos) {
    uint *Oid1;
    uint *X1;
    uint *Y1;
    uint m1 = pow(K, J->ktree->maxLevel + 1) - 1;
    uint n1;
    // Llama a la función rangeQueryHDOriginal para obtener los puntos
    rangeQueryHDOriginal(J, 0, m1, 0, m1, Oid1, X1, Y1, n1);
    Point *Ar = new Point[nelementos];
    // Llena el vector con los puntos obtenidos
    for (uint i = 0; i < n1; i++) {
        Point p1;
        p1.setX(X1[i]);
        p1.setY(Y1[i]);
        Ar[i] = p1;
    }
    // Retorna el vector de puntos
    return Ar;
}

/*arreglado por fernando santolaya... para experimentos..*/

double abdelHDD2( std::vector<Point>& Er,  std::vector<Point>& Br) {
    double cmax = 0.0, cmin, d;

    int largoEr = Er.size();
    //int largoBr = Br.size();

    for (int i = 0; i < largoEr; i++) {
        cmin = std::numeric_limits<double>::max();
        for (int j = 0; j < largoEr; j++) {
            d = (Er[i].getX() - Br[j].getX()) * (Er[i].getX() - Br[j].getX()) +
                (Er[i].getY() - Br[j].getY()) * (Er[i].getY() - Br[j].getY());

            if (d < cmin) cmin = d;
            if (d <= cmax) break; // Early break
        }
        if (cmin > cmax) cmax = cmin;
    }

    return sqrt(cmax);
}
double hausdorffDistTaha2( vector<Point> &A,  vector<Point> &B){
    return abdelHDD2(A, B);
}


double hausdorffDistTaha(Snapshot *J, Snapshot *H, int nelementos) {
    uint *Oid1, *Oid2;
    uint *X1, *X2;
    uint *Y1, *Y2;
    uint m1 = pow(K, J->ktree->maxLevel + 1) - 1;
    uint m2 = pow(K, H->ktree->maxLevel + 1) - 1;
    uint n1, n2;
    rangeQueryHDOriginal(J, 0, m1, 0, m1, Oid1, X1, Y1, n1);
    rangeQueryHDOriginal(H, 0, m2, 0, m2, Oid2, X2, Y2, n2);
    //cout << m << " fsdf " << n << endl;
    double cmax = 0.0;
    Point p1, p2;
    Point *Ar = new Point[nelementos];
    Point *Br = new Point[nelementos];
    for (uint i = 0; i < n1; i++) {
        p1.setX(X1[i]);
        p1.setY(Y1[i]);
        //cout << p1.getX() << "," << p1.getY() << endl;
        Ar[i] = p1;
    }
    for (uint i = 0; i < n1; i++) {
        //cout << "Punto:" << X[i] << endl;
        //nnMax(B, Point(X[i], Y[i]), cmax);
        //cout << "Elementos: " << n << endl;

        p2.setX(X2[i]);
        p2.setY(Y2[i]);
        Br[i] = p2;
    }
    cmax = abdelHDD(Ar, Br, nelementos);
    delete[] Oid1;
    delete[] X1;
    delete[] Y1;
    delete[] Oid2;
    delete[] X2;
    delete[] Y2;
    return cmax;
}

double naiveHDD(Point p1[], Point p2[], int largo) {
    double d = 0.0;
    double cmin = 0.0;
    double cmax = 0.0;
    //for (Point x : p1) {
    //Point x = new Point();
    //Point y = new Point();
    for (int i = 0; i < largo; i++) {
        cmin = std::numeric_limits<double>::max();
        //for (Point y : p2) {
        for (int j = 0; j < largo; j++) {
            d = (p1[i].getX() - p2[j].getX()) * (p1[i].getX() - p2[j].getX()) +
                (p1[i].getY() - p2[j].getY()) * (p1[i].getY() - p2[j].getY());
            //distancia = sqrt((x1 - y1)*(x1 - y1 ) + (x2 - y2)*(x2 - y2));
            //d = p1[i].restar(p2[j]);
            //cout << d << ",";
            if (d < cmin) cmin = d;
        }
        //cout << endl;
        if (cmin > cmax) cmax = cmin;
    }
    delete[] p1;
    delete[] p2;
    return sqrt(cmax);
}

double hausdorffDistNaive(Snapshot *L, Snapshot *P, int nelementos) {
    uint *Oid1, *Oid2;
    uint *X1, *X2;
    uint *Y1, *Y2;
    uint m1 = pow(K, L->ktree->maxLevel + 1) - 1;
    uint m2 = pow(K, P->ktree->maxLevel + 1) - 1;
    uint n1, n2;
    rangeQueryHDOriginal(L, 0, m1, 0, m1, Oid1, X1, Y1, n1);
    rangeQueryHDOriginal(P, 0, m2, 0, m2, Oid2, X2, Y2, n2);
    //cout << m << " fsdf " << n << endl;
    double cmax = 0.0;
    Point p1, p2;
    Point *Ar = new Point[nelementos];
    Point *Br = new Point[nelementos];
    for (uint i = 0; i < n1; i++) {
        p1.setX(X1[i]);
        p1.setY(Y1[i]);
        //cout << p1.getX() << "," << p1.getY() << endl;
        Ar[i] = p1;
    }
    for (uint i = 0; i < n1; i++) {
        //cout << "Punto:" << X[i] << endl;
        //nnMax(B, Point(X[i], Y[i]), cmax);
        //cout << "Elementos: " << n << endl;

        p2.setX(X2[i]);
        p2.setY(Y2[i]);
        Br[i] = p2;
    }
    cmax = naiveHDD(Ar, Br, nelementos);
    delete[] Oid1;
    delete[] X1;
    delete[] Y1;
    delete[] Oid2;
    delete[] X2;
    delete[] Y2;
    return cmax;

}
//*********************************************************************************************

//entrada.
// rep: un snapshot
// q: el punto de referencia
//salida:
//	oid: el identificador del punto más cercano
//  p: la posición del punto más cercanco.
void NN(Snapshot *rep, Point q, uint &oid, Point &p) {
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    ElementQueue e, c;
    e.sTree = 0;
    e.area.x1 = 0;
//el 2 es porque k = 2, reemplazar por K.
    e.area.x2 = pow(K, rep->ktree->maxLevel) - 1;
    e.area.y1 = e.area.x1;
    e.area.y2 = e.area.x2;
    e.priority = maxDist(q, e.area);
    c = e;
    c.priority = e.priority + 1;        //explicar el + 1 con un comentario!!!!
    int foundNN = 0;
    long int subTreeIndex;
    double minDis2Child;
    pQueue.push(e);
    while (!pQueue.empty()) {
        e = pQueue.top();
        pQueue.pop();
        if (isLeaf(rep->ktree, e.sTree)) {
            //si es hoja de último nivel
            if ((e.priority < c.priority)) {
                c = e;
                foundNN = 1;
                //Al encontrar un candidato, puede haber otro mejor
                //en otra área, hay que seguir buscando!
            }
        } else {
            subTreeIndex = firstChild(rep->ktree, e.sTree);
            //child function return -1 if sTree no have any child
            //no tiene hijo si en la posición e.sTreee hay un 0.
            if (subTreeIndex != -1) {
                for (int i = 0; i < (K * K); i++) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = maxDist(q, child.area);
                    minDis2Child = minDist(q, child.area);
                    if (minDis2Child < c.priority) {
                        pQueue.push(child);
                    }
                }
            }
        }
    }
//hay que obtener el ID de c.
    if (foundNN) {
        p.setX(c.area.x1);   //en la hoja x1=x2
        p.setY(c.area.y1);   //en la hoja y1=y2
        uint len = 1;
        int rankLeaf = rep->ktree->bl->rank1(posInLeaf(rep->ktree, c.sTree));
        uint *resp = rep->labels->getUpToKObjects(rankLeaf, len);
        //está garantizado que hay por lo menos un objeto en la celda por las
        //propiedades del K2-tree.
        oid = resp[0];
        delete[] resp;
        resp = NULL;
    } else {
        oid = -1;
    }

}

//Filtros des vecinos más cercanos.
/*
 * Si la consulta de vecinos más cercanos cae en un instante que no tiene
 * snapshot entonces es necesario realizar la consutalta en el snapshot más
 * cercano.  Esto tiene como consecuencia que no se pueda determinar en el
 * snapshot los vecinos más cercanos, sino que entregar una lista de candidatos
 * posibles, los cuales deben ser evaluados posteriormente, luego de actualizar
 * su posición al instante de la consulta.  Para la realización del filtro
 * tanto en las consultas del vecino más cercano y la de los k vecinos más
 * cercanos se necesita incorporar un parámetro que corresconde a la distancia
 * máxima que un objeto se puede desplazar desde el snapshot al instante de la
 * consulta.  La distancia está dada en número de celdas que es la unidad mínima
 * de distancia en nuestro modelo.
 *
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/*
 * oid y p son arreglos no nulos de tamaño kn
 */
void kNNFilter(Snapshot *rep, Point q, uint maxDesp, uint &kn, uint *oid,
               Point *p) {
    priority_queue<CandidateElement, std::vector<CandidateElement>,
            maxHeapComparator> candidates;
    priority_queue<ElementQueue, std::vector<ElementQueue>, minHeapComparator> pQueue;
    ElementQueue e;
    CandidateElement ce;
    e.sTree = 0;
    e.area.x1 = 0;
    e.area.x2 = pow(2, rep->ktree->maxLevel) - 1;
    e.area.y1 = e.area.x1;
    e.area.y2 = e.area.x2;
    e.priority = maxDist(q, e.area);
    long int subTreeIndex;
    double minDis2Child;
    long uint lenCandidates = 0;
    uint numObj = 0;
    uint toAdd = 0;
    uint len = 0;
    uint *resp;
    int rankLeaf;
    pQueue.push(e);
    while (!pQueue.empty()) {
        e = pQueue.top();
        pQueue.pop();
        if (isLeaf(rep->ktree, e.sTree)) {
            //es una hoja de 'ultimo nivel
            //aquí la hoja entera entra a los candidatos o se descarta.
            //por lo tanto la recuperación de los objetos candidatos se deja
            //para el final.

        } else {
            subTreeIndex = firstChild(rep->ktree, e.sTree);
            if (subTreeIndex != -1) {
                //no es una hoja intermedia, es decir una región sin objetos
                for (int i = 0; i < (K * K); i++) {
                    ElementQueue child;
                    child.sTree = subTreeIndex + i;
                    child.area = getSubArea(e.area, i);
                    child.priority = maxDist(q, child.area);
                    minDis2Child = minDist(q, child.area);

                    if (!((candidates.size() >= kn)
                          && (minDis2Child >= candidates.top().priority))) {
                        pQueue.push(child);
                    }
                }
            }
        }
    }
    kn = candidates.size();
//puede ser que existan menos kn que los originalmente pedidos
//entonces en la respuesta kn serán los realmente obtenidos.
//lo siguiente es un overhead que se podría evitar si se retorna la cola
//directamente.
    uint i = kn - 1;
    while (!candidates.empty()) {
        ce = candidates.top();
        oid[i] = ce.oid;
        p[i] = Point(ce.x, ce.y);
        i--;
        candidates.pop();
    }
}

//////////////////////////////////////////////////////////////////////////////
size_t sizeMREP2(MREP2 *rep) {
    size_t totalByte = 0;
    if (rep != NULL) {
        totalByte = sizeof(MREP2);
        totalByte += sizeof(long long uint) * (rep->maxLevel + 1);
        totalByte += rep->bl->getSize();
        totalByte += rep->bn->getSize();
        totalByte += rep->bt->getSize();
    }
    return totalByte;
}

size_t sizeSnapshot(Snapshot *lktrep) {
    size_t tam = 0;
    if (lktrep != NULL) {
        tam += sizeof(Snapshot);
        tam += sizeMREP2(lktrep->ktree);
        //tam += lktrep->labels->getSize();
    }
    return tam;
}


int getHeapMaxElements(){
    return heapMaxElements;
}
int getHeapExitElements(){
    return exitHeapElements;
}

