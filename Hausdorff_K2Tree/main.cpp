//
// Created by fernando-lenovo on 09-09-24.
//
#include <stdio.h>
#include <stdlib.h>
#include "ryu-kamata.h"
#include <string.h>
#include "Util/TimeMesure.h"
#include <dirent.h>
#include "utils.h"
#include<BitSequence.h>
#include <Snapshot.h>
#include "MemoryMeasure.h"
using namespace std;

std::string extractFilename(const std::string &path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return path; // Path doesn't contain a '/' separator
    } else {
        return path.substr(pos + 1); // Extract filename from position after '/'
    }
}
std::string extractCategoryName(const std::string &path) {
    // Encuentra la última ocurrencia de "sets"
    size_t startPos = path.find("sets");
    if (startPos != std::string::npos) {
        startPos += 0; // Mueve el puntero después de "sets"

        // Encuentra la siguiente barra '/' después de "sets"
        size_t endPos = path.find('/', startPos);
        if (endPos != std::string::npos) {
            return path.substr(startPos, endPos - startPos);
        } else {
            // Si no se encuentra '/', toma el resto de la cadena
            return path.substr(startPos);
        }
    }
    return ""; // Retorna una cadena vacía si "sets" no se encuentra
}
std::string extractDistribution(const std::string &path) {
    size_t pos = path.find("Conjunto");
    if (pos != std::string::npos) {
        // Encuentra la posición del último número después de "Conjunto"
        pos = path.find_first_of("0123456789", pos);
        if (pos != std::string::npos && pos + 1 < path.length()) {
            char distChar = path[pos + 1]; // Captura el carácter justo después del número

            // Retorna el nombre completo de la distribución según la letra
            switch (distChar) {
                case 'R':
                    return "RANDOM";
                case 'M':
                    return "MIXTA";
                case 'G':
                    return "GAUSS";
                case 'T':
                    return "REAL";
                default:
                    return "UNKNOWN"; // En caso de que no sea R, M o G
            }
        }
    }
    return "UNKNOWN"; // Retorna "UNKNOWN" si no se encuentra la distribución
}
void printHeader() {
    printf("%-19s|%-8s|%-12s|%-12s|"
           "%-15s|%-15s|%-15s|%-15s|"
           "%-12s|%-7s|%-15s|%-15s|%-15s|%-15s|%-15s|\n",
           "SET NAME", "CATEGORY", "ALGORITHM", "DISTRIBUTION",
           "BUILD TIME", "QUERY TIME", "QUERY WALL TIME", "EXTRACT TIME",
           "Haus Dist", "LAMBDA", "STORAGE", "PEAK REAL MEM", "PEAK VIRT MEM",
           "MXHEAP", "EXIT HEAP"); // Añadir nuevos encabezados
}

void printStats(const Stats &stats) {
    printf("%-19s|%-8s|%-12s|%-12s|%15.01f|%15.01f|%15.01f|%15.01f|%12.01f|%7d|%15.01f|%15lu|%15lu|%15d|%15d|\n",
           stats.dataset.c_str(), stats.datasetCategory.c_str(),
           stats.algoritmo.c_str(), stats.distribution.c_str(),
           stats.buildTime, stats.queryTime, stats.queryTimeWall, stats.extractTime,
           stats.hauss_distance, stats.lambda, stats.storage,
           stats.peakRealMem, stats.peakVirtMem, stats.mxheap, stats.exit_heap); // Imprimir nuevos parámetros
}
int cuentaPuntos(string ruta1) {
    ifstream set1;
    set1.open(ruta1);
    string linea1;
    int nelementos1 = 0;
    //Necesitamos saber cuantos elementos tiene el
    while (getline(set1, linea1)) {
        nelementos1++;
    }
    //cout << nelementos1 << endl;
    set1.close();
    return nelementos1;
}
lkt *llenaK2tree(string ruta) {
    lkt *tmp = createLKTree(3); //16 ERA ANTES
    string linea1;
    ifstream set1;
    set1.open(ruta);
    int z , i = 0;
    string token;
    int aux[2];
    while (getline(set1, linea1)) {
        //Ahora leemos la linea y obtenemos los puntos
        istringstream ss(linea1);
        z = 0;
        while (getline(ss, token, ',')) {
            aux[z] = atoi(token.c_str());
            z++;
        }
        insertNode(tmp, aux[0], aux[1], i);
        i++;
    }
    set1.close();

    return tmp;
}
void imprimeResultados(Stats & resultados){
    char *homedir = getenv("HOME");
    strcat(homedir, "/RESULTADOS/resultHauss.txt");
    if (checkFileExist(homedir) == 1 && fileIsEmpty(homedir) == 1) {
        printHeader();
        fflush(stdout);
    }
    printStats(resultados);
    fflush(stdout);

}
void printPoints(Point* points, int count) {
    for (int i = 0; i < count; ++i) {
        points[i].print();
    }
}
void HDK2(Snapshot *snapsp, Snapshot *snapsp2, Stats &resultados ){
    hausdorffDistHDK2(snapsp, snapsp2);
   /* Cronometer *crono = cCronometer();
    double timehdk2 = 0.0;
    start_clock(crono);
    auto haussCompletohdk2 = std::max(hausdorffDistHDK2(snapsp, snapsp2), hausdorffDistHDK2(snapsp2, snapsp));
    timehdk2 = (double) (stop_clock(crono) * 1000000.0);
    resultados.algoritmo = "K2T-HDK2-p1";
    resultados.hauss_distance = haussCompletohdk2;
    resultados.queryTime = timehdk2;
    resultados.queryTimeWall =  wallTime(crono)* 1000000.0;
    resultados.storage= sizeSnapshot(snapsp) + sizeSnapshot(snapsp2);
    resultados.lambda = 0; //no ocupa.
    resultados.peakRealMem = getResidentMemory();
    resultados.peakVirtMem = getVirtualMemory();
    imprimeResultados(resultados);
    */
}
void K2TMAXHEAP(Snapshot *snapsp, Snapshot *snapsp2, Stats &resultados){
    Cronometer *crono = cCronometer();
    double time = 0.0;
    start_clock(crono);
    auto haussCompleto = std::max( hausdorffDistHDK3MaxHeap(snapsp, snapsp2), hausdorffDistHDK3MaxHeap(snapsp2, snapsp));
    time = (double) (stop_clock(crono) * 1000000.0);
    resultados.algoritmo = "K2T-MXHEAP";
    resultados.hauss_distance = haussCompleto;
    resultados.queryTime = time;
    resultados.queryTimeWall =  wallTime(crono)* 1000000.0;
    resultados.storage= sizeSnapshot(snapsp) + sizeSnapshot(snapsp2);
    resultados.hauss_distance = haussCompleto;
    resultados.lambda = 0;
    resultados.peakRealMem = getResidentMemory();
    resultados.peakVirtMem = getVirtualMemory();
    resultados.exit_heap = getHeapExitElements();
    resultados.mxheap = getHeapMaxElements();
    imprimeResultados(resultados);
}
void K2TMAXHEAPv2(Snapshot *snapsp, Snapshot *snapsp2, Stats &resultados){
    Cronometer *crono = cCronometer();
    double time = 0.0;
    start_clock(crono);
    auto haussCompleto = std::max( hausdorffDistHDK3MaxHeapv2(snapsp, snapsp2), hausdorffDistHDK3MaxHeapv2(snapsp2, snapsp));
    time = (double) (stop_clock(crono) * 1000000.0);
    resultados.algoritmo = "K2T-MXHEAPv2";
    resultados.hauss_distance = haussCompleto;
    resultados.queryTime = time;
    resultados.queryTimeWall =  wallTime(crono)* 1000000.0;
    resultados.storage= sizeSnapshot(snapsp) + sizeSnapshot(snapsp2);
    resultados.hauss_distance = haussCompleto;
    resultados.lambda = 0;
    resultados.peakRealMem = getResidentMemory();
    resultados.peakVirtMem = getVirtualMemory();
    resultados.exit_heap = getHeapExitElements();
    resultados.mxheap = getHeapMaxElements();
    imprimeResultados(resultados);
}
void KAMATA(Snapshot *snapsp, Snapshot *snapsp2, Stats &resultados, int cantPuntosSet1, int cantPuntosSet2){
    Cronometer *crono = cCronometer();
    double extrTime = 0.0;
    start_clock(crono);
    auto dataset1 = extractPointK2tree(snapsp, cantPuntosSet1);
    auto dataset2 = extractPointK2tree(snapsp2, cantPuntosSet2);
    extrTime = (double) (stop_clock(crono) * 1000000.0);

    double time = 0.0;
    start_clock(crono);
    auto haussCompleto = hausKamata(dataset1,dataset2, 3);
    time = (double) (stop_clock(crono) * 1000000.0);

    resultados.algoritmo = "K2T-KAMATA";
    resultados.queryTime = time;
    resultados.queryTimeWall =  wallTime(crono)* 1000000.0;
    resultados.extractTime = extrTime;
    resultados.storage= (((sizeof(int)*2)*cantPuntosSet1) + ((sizeof(int)*2)*cantPuntosSet2)) + (sizeof(double)*cantPuntosSet1 + sizeof(double)*cantPuntosSet2) ;
    resultados.hauss_distance = haussCompleto;
    resultados.lambda = 3;
    resultados.peakRealMem = getResidentMemory();
    resultados.peakVirtMem = getVirtualMemory();
    imprimeResultados(resultados);
}
void TAHA(Snapshot *snapsp,Snapshot *snapsp2, Stats &resultados, int cantPuntosSet1, int cantPuntosSet2){
    Cronometer *crono = cCronometer();
    double extrTime = 0.0;
    start_clock(crono);
    auto dataset1 = extractPointK2tree(snapsp, cantPuntosSet1);
    auto dataset2 = extractPointK2tree(snapsp2, cantPuntosSet2);
    extrTime = (double) (stop_clock(crono) * 1000000.0);

    double time = 0.0;
    start_clock(crono);
    auto haussCompleto = std::max(hausdorffDistTaha2(dataset1, dataset2), hausdorffDistTaha2(dataset2, dataset1));
    time = (double) (stop_clock(crono) * 1000000.0);
    resultados.algoritmo = "K2T-TAHA";
    resultados.hauss_distance = haussCompleto;
    resultados.queryTime = time;
    resultados.queryTimeWall = wallTime(crono)* 1000000.0;
    resultados.extractTime = extrTime;
    resultados.storage= ((sizeof(int)*2)*cantPuntosSet1)+ ((sizeof(int)*2)*cantPuntosSet2);
    resultados.hauss_distance = haussCompleto;
    resultados.lambda = 0;
    resultados.peakRealMem = getResidentMemory();
    resultados.peakVirtMem = getVirtualMemory();
    imprimeResultados(resultados);
}


void ejecutaExperimentos(const std::string &dataset1, const std::string &dataset2, int experimento) {
    Cronometer *crono = cCronometer();
    Stats resultados; // para guardar el resultado de los experimentos.

    int cantPuntosSet1 = cuentaPuntos(dataset1);
    int cantPuntosSet2 = cuentaPuntos(dataset2);

    double buildTime = 0.0;
   // int tamMatrix = pow(2, 16);
    Snapshot *snapsp = nullptr;
    Snapshot *snapsp2 = nullptr;

    // Aquí leerías el Snapshot desde el archivo
    std::string finalFilename1 = extractFilename(dataset1) + ".kt";
    std::string finalFilename2 = extractFilename(dataset2) + ".kt";


    if (loadSnapshotFromFile(&snapsp, finalFilename1.c_str()) != 0) {
        std::cerr << "Error al cargar snapshot 1." << std::endl;
    }

    if (loadSnapshotFromFile(&snapsp2, finalFilename2.c_str()) != 0) {
        std::cerr << "Error al cargar snapshot 2." << std::endl;
    }

    resultados.dataset = extractFilename(dataset1);
    resultados.datasetCategory = extractCategoryName(dataset1);
    resultados.distribution = extractDistribution(resultados.dataset);
    resultados.buildTime = buildTime;

    // Ejecución del experimento según el tipo especificado
    switch (experimento) {
        case 0:
            HDK2(snapsp, snapsp2, resultados);
            break;
        case 1:
            K2TMAXHEAP(snapsp, snapsp2, resultados);
            break;
        case 2:
            KAMATA(snapsp, snapsp2, resultados, cantPuntosSet1, cantPuntosSet2);
            break;
        case 3:
            TAHA(snapsp, snapsp2, resultados, cantPuntosSet1, cantPuntosSet2);
            break;
        case 4:
            K2TMAXHEAPv2(snapsp, snapsp2, resultados);
            break;
        default:
            std::cerr << "Experimento desconocido." << std::endl;
            return;
    }

    // Liberar recursos
    destroySnapshot(snapsp);  // Asegúrate de implementar esta función
    destroySnapshot(snapsp2); // Asegúrate de implementar esta función
    resultados.reset();
}
void construirEstructura(const std::string &dataset1, const std::string &dataset2){
   // int tamMatrix = pow(2, 16);
   int tamMatrix = pow (2,4); // 2,19 para reales
    Snapshot *snapsp = nullptr;
    Snapshot *snapsp2 = nullptr;
    int cantPuntosSet1 = cuentaPuntos(dataset1);
    int cantPuntosSet2 = cuentaPuntos(dataset2);

    lkt *tmp = llenaK2tree(dataset1);
    lkt *tmp2 = llenaK2tree(dataset2);
    snapsp = createSnapshot(tmp, tamMatrix, cantPuntosSet1, cantPuntosSet1);
    snapsp2 = createSnapshot(tmp2, tamMatrix, cantPuntosSet1, cantPuntosSet2);

    std::string snap1filename = extractFilename(dataset1) + ".kt";
    std::string snap2filename = extractFilename(dataset2) + ".kt";
    saveSnapshotToFile(snapsp, snap1filename.c_str());
    saveSnapshotToFile(snapsp2, snap2filename.c_str());
}

int main(int argc, char **argv) {

    printHeader();
    /*if (argc < 4) {
        printf("%s <PATH DATASETS> <EXPERIMENTO> <CONSTRUIR (1/0)>\n", argv[0]);
        printf("EXPERIMENTO: 0 (HDK2), 1 (K2TMAXHEAP), 2 (KAMATA), 3 (TAHA)\n");
        return -1;
    }
    // Inicializar GEOS
    std::string dataset1 = argv[1];
    std::string dataset2 = argv[2];

    int experimentoSeleccionado = std::stoi(argv[3]);
    int construir = std::stoi(argv[4]); // 1 para construir, 0 para no construir

    if (construir == 1) {
        construirEstructura(dataset1, dataset2);
    } else {
        ejecutaExperimentos(dataset1, dataset2, experimentoSeleccionado);
    }
*/
    return 0;
}
