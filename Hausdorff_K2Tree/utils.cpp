//
// Created by fernando on 05-07-24.
//
 #include "utils.h"
#include <cstring>
#include <sys/stat.h>
#include <dirent.h>
#include <stdlib.h>
#include <iostream>
#include <random>


int checkFileExist(char file[]){
    struct stat st;
    return !stat(file, &st);
}
unsigned long long getFileSize(std::string file) {
    struct stat st;
    int status = stat(file.c_str(), &st);

    if (status == 0) {
        return static_cast<unsigned long long>(st.st_size); // Tamaño en bytes
    } else {
        // Manejar el error, por ejemplo, devolver 0 o lanzar una excepción
        return 0;
    }
}
int fileIsEmpty(char file[]){
    FILE *fp = fopen( file,"r" );
    int c = getc(fp);
    if (c == EOF)
        return 1;
    ungetc(c, fp);
    return 0;

}
int findQuantityOfDatasets(int argc, char *argv[]) {
    //Using the argv[5](folder) to find the quantity of datasets
    DIR *dir;
    struct dirent *ent;
    int quantityOfDatasets = 0;
    if ((dir = opendir(argv[5])) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_type == DT_REG) {
                quantityOfDatasets++;
            }
        }
        closedir(dir);
        return quantityOfDatasets;
    } else {
        perror("No se pudo abrir el directorio");
        exit(EXIT_FAILURE);
    }

}
char *recoverPath(int argc, char *argv[], int i) {
    DIR *dir;
    struct dirent *ent;
    int quantityOfDatasets = 0;
    if ((dir = opendir(argv[5])) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_type == DT_REG) {
                if (quantityOfDatasets == i) {
                    char *path = (char *) malloc(sizeof(char) * 100);
                    strcpy(path, argv[5]);
                    // strcat(path, "/");
                    strcat(path, ent->d_name);
                    return path;
                }
                quantityOfDatasets++;
            }
        }
        closedir(dir);
    } else {
        perror("No se pudo abrir el directorio");
        exit(EXIT_FAILURE);
    }
    return NULL;
}
double convertByteToKb(double bytes){
    return bytes / CONVERSION_VALUE;
}
std::string removeExtension(std::string fullname){
    size_t lastindex = fullname.find_last_of(".");
    std::string rawname = fullname.substr(0, lastindex);
    return rawname;
}
/*double distance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}*/
int generarNumeroAleatorio(int min, int max) {
    // Configurar un generador de números aleatorios usando el motor Mersenne Twister
    std::random_device rd; // Obtener un seed de random_device
    std::mt19937 gen(rd()); // Mersenne Twister 19937 como generador

    // Definir una distribución uniforme entre min y max (ambos inclusivos)
    std::uniform_int_distribution<> distribucion(min, max);

    // Generar y devolver el número aleatorio
    return distribucion(gen);
}


