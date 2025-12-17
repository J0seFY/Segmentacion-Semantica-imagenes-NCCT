
#ifndef MEMORY_MEASURE_H_
#define MEMORY_MEASURE_H_
/** Based on code by
 *
 * Tyson Jones 2017
 * tyson.jones@materials.ox.ac.uk
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Measures the current (and peak) resident and virtual memory
 * usage of your linux C process, in bytes, accurate to nearest kB.
 * Returns a 0 if memory info access was successful, else prints
 * an error message and returns 1
 */
int getMemory(
        unsigned long *currRealMem, unsigned long *peakRealMem,
        unsigned long *currVirtMem, unsigned long *peakVirtMem) {

    // stores each word in status file
    char buffer[1024] = "";

    // linux file contains this-process info
    FILE* file = NULL;
    file = fopen("/proc/self/status", "r");
    if (file == NULL) {
        printf("Call to getMemory FAILED; "
               "linux file proc/self/status not found!\n");
        return 1;
    }

    // read the entire file, recording mems in kB
    while (fscanf(file, " %1023s", buffer) == 1) {

        if (strcmp(buffer, "VmRSS:") == 0) {
            fscanf(file, " %lu", currRealMem);
        }
        if (strcmp(buffer, "VmHWM:") == 0) {
            fscanf(file, " %lu", peakRealMem);
        }
        if (strcmp(buffer, "VmSize:") == 0) {
            fscanf(file, " %lu", currVirtMem);
        }
        if (strcmp(buffer, "VmPeak:") == 0) {
            fscanf(file, " %lu", peakVirtMem);
        }
    }
    fclose(file);

    // convert kB to bytes
//    unsigned int factor = 1000;
//    *currRealMem *= factor;
//    *peakRealMem *= factor;
//    *currVirtMem *= factor;
//    *peakVirtMem *= factor;

    return 0;
}

void print_memory_usage(){
    unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
    getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
    std::cout<<"Peak memory usage: Resident => " <<peakRealMem << " KB; Virtual => " << peakVirtMem << " KB" <<std::endl;
}
unsigned long getResidentMemory(){
    unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
    getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
    return peakRealMem;
}
unsigned long getVirtualMemory(){
    unsigned long currRealMem, peakRealMem, currVirtMem, peakVirtMem;
    getMemory(&currRealMem, &peakRealMem, &currVirtMem, &peakVirtMem);
    return peakVirtMem;
}

#endif // MEMORY_MEASURE_H_


