//
// Created by fernando on 23-11-22.
//
#ifndef UTIL_H
#define UTIL_H

#include <string>

typedef struct Stats {
    std::string dataset;
    std::string algoritmo;
    std::string distribution;
    std::string datasetCategory;

    double buildTime;
    double queryTime;
    double queryTimeWall;
    double extractTime;
    double hauss_distance;
    int lambda;
    double storage;
    unsigned long peakRealMem;
    unsigned long peakVirtMem;

    // Nuevos parámetros como enteros
    int mxheap;
    int exit_heap;

    Stats()
            : dataset(""), algoritmo(""), distribution(""), datasetCategory(""),
              buildTime(0.0), queryTime(0.0), extractTime(0.0), queryTimeWall(0.0),
              hauss_distance(0.0), lambda(0), storage(0.0), peakRealMem(0), peakVirtMem(0),
              mxheap(0), exit_heap(0) {} // Inicialización de nuevos parámetros

    void reset() {
        algoritmo = "";
        queryTime = 0.0;
        extractTime = 0.0;
        hauss_distance = 0.0;
        lambda = 0;
        storage = 0.0;
        peakVirtMem = 0;
        peakRealMem = 0;
        mxheap = 0; // Reiniciar nuevo parámetro
        exit_heap = 0; // Reiniciar nuevo parámetro
        queryTimeWall = 0.0;
    }
} Stats;

const int CONVERSION_VALUE = 1024;

int checkFileExist(char file[]);
int fileIsEmpty(char file[]);
unsigned long long getFileSize(std::string file);
int generarNumeroAleatorio(int min, int max);

#endif // UTIL_H
