// debug_kamata.cpp
// Harness de estrés para reproducir corrupción de memoria en hausKamata
// Uso: DebugKamata <ruta_carpeta_txt> <cantidad_pares_a_probar>
// Procesa pares (0.txt,1.txt), (2.txt,3.txt), ... hasta el número indicado.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <chrono>

#include "ryu-kamata.h"      // prototipo hausKamata(const std::vector<Point>&, const std::vector<Point>&, int)
#include "Util/Point.h"      // clase Point(int x, int y)

static std::vector<Point> leerPuntos(const std::string &ruta) {
    std::vector<Point> puntos;
    std::ifstream in(ruta);
    if (!in.is_open()) {
        std::cerr << "[WARN] No se pudo abrir archivo: " << ruta << '\n';
        return puntos; // vacío
    }
    std::string linea;
    linea.reserve(64);
    while (std::getline(in, linea)) {
        if (linea.empty()) continue;
        int x, y; char coma;
        std::istringstream iss(linea);
        if (!(iss >> x >> coma >> y)) {
            continue; // línea inválida
        }
        puntos.emplace_back(x, y);
    }
    return puntos;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Uso: " << argv[0] << " <ruta_carpeta_txt> <cantidad_pares_a_probar>\n";
        return 1;
    }

    const std::string carpeta = argv[1];
    const int cantidadPares = std::atoi(argv[2]);
    if (cantidadPares <= 0) {
        std::cerr << "[ERROR] cantidad_pares_a_probar debe ser > 0\n";
        return 1;
    }

    std::cout << "[INFO] Iniciando harness Kamata. Carpeta: " << carpeta
              << " | Pares a procesar: " << cantidadPares << '\n';

    for (int i = 0; i < cantidadPares; ++i) {
        const int idxA = 2 * i;
        const int idxB = 2 * i + 1;
        std::string rutaA = carpeta + "/" + std::to_string(idxA) + ".txt";
        std::string rutaB = carpeta + "/" + std::to_string(idxB) + ".txt";

        auto puntosA = leerPuntos(rutaA);
        auto puntosB = leerPuntos(rutaB);

        if (puntosA.empty() || puntosB.empty()) {
            std::cout << "[PAIR " << i << "] Archivos vacíos o faltantes (" << idxA << ", " << idxB << "). Saltando.\n";
            continue;
        }

        double distancia = 0.0;
        try {
            distancia = hausKamata(puntosA, puntosB, 3); // k=3 fijo
        } catch (const std::exception &ex) {
            std::cerr << "[PAIR " << i << "] Excepción en hausKamata: " << ex.what() << '\n';
            continue;
        }

        std::cout << "[PAIR " << i << "] Resultado Kamata (" << idxA << " vs " << idxB
                  << "): " << distancia << " | puntosA=" << puntosA.size()
                  << " puntosB=" << puntosB.size() << '\n';
    }

    std::cout << "[INFO] Finalizado harness Kamata.\n";
    return 0;
}
