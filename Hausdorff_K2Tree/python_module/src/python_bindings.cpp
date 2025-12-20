#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>

#include "../../Snapshot/Snapshot.h"
#include "../../ryu-kamata.h"
#include "../../utils.h"
#include "../../Util/Point.h"


namespace py = pybind11;

// Función auxiliar para convertir máscara numpy a vector de puntos - ELABORACION PROPIA
std::vector<Point> mask_to_points(py::array_t<uint8_t> mask) {
    auto buf = mask.request();
    
    if (buf.ndim != 2) {
        throw std::runtime_error("La máscara debe ser 2D");
    }
    
    uint8_t *ptr = (uint8_t *) buf.ptr;
    int height = buf.shape[0];
    int width = buf.shape[1];
    
    std::vector<Point> points;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (ptr[y * width + x] > 0) {  // Pixel activo (valor 1)
                Point p;
                p.x = x;
                p.y = y;
                points.push_back(p);
            }
        }
    }
    
    return points;
}

// Función auxiliar para crear K2Tree desde puntos
lkt* points_to_k2tree(const std::vector<Point>& points, int max_level = 10) {
    lkt *tree = createLKTree(max_level);
    
    for (size_t i = 0; i < points.size(); i++) {
        insertNode(tree, points[i].x, points[i].y, i);
    }
    
    return tree;
}

// Función auxiliar para crear Snapshot desde K2Tree
Snapshot* create_snapshot_from_points(const std::vector<Point>& points, int max_level = 10) {
    lkt* tree = points_to_k2tree(points, max_level);
    
    // Calcular tamaño de matriz basado en max_level
    int tam_matrix = std::pow(2, max_level);
    
    Snapshot* snapshot = createSnapshot(tree, tam_matrix, points.size(), points.size());
    
    // Limpiar el árbol temporal
    destroyLKTree(tree);
    
    return snapshot;
}

// Clase wrapper para facilitar el uso
class HausdorffCalculator {
public:
    HausdorffCalculator(int max_level = 10) : max_level_(max_level) {}
    
    // Algoritmo K2T-MAXHEAP
    double calculate_k2t_maxheap(py::array_t<uint8_t> mask1, py::array_t<uint8_t> mask2) {
        auto points1 = mask_to_points(mask1);
        auto points2 = mask_to_points(mask2);
        
        if (points1.empty() || points2.empty()) {
            return 0.0;  // Si alguna máscara está vacía
        }
        
        Snapshot* snap1 = create_snapshot_from_points(points1, max_level_);
        Snapshot* snap2 = create_snapshot_from_points(points2, max_level_);
        
        double dist1 = hausdorffDistHDK3MaxHeap(snap1, snap2);
        double dist2 = hausdorffDistHDK3MaxHeap(snap2, snap1);
        double hausdorff_dist = std::max(dist1, dist2);
        
        destroySnapshot(snap1);
        destroySnapshot(snap2);
        
        return hausdorff_dist;
    }
    
    // Algoritmo K2T-MAXHEAPv2
    double calculate_k2t_maxheap_v2(py::array_t<uint8_t> mask1, py::array_t<uint8_t> mask2) {
        auto points1 = mask_to_points(mask1);
        auto points2 = mask_to_points(mask2);
        
        if (points1.empty() || points2.empty()) {
            return 0.0;
        }
        
        Snapshot* snap1 = create_snapshot_from_points(points1, max_level_);
        Snapshot* snap2 = create_snapshot_from_points(points2, max_level_);

        // Importante: calcular la versión SIMÉTRICA como el máximo de las dos
        // distancias dirigidas de v2, para evitar efectos de estado compartido
        // (supermax) que podrían alterar el resultado si se reutiliza entre llamadas.
        double d1 = hausdorffDistHDK3MaxHeapv2(snap1, snap2);
        double d2 = hausdorffDistHDK3MaxHeapv2(snap2, snap1);
        double hausdorff_dist = std::max(d1, d2);

        destroySnapshot(snap1);
        destroySnapshot(snap2);
        
        return hausdorff_dist;
    }
    
    // Algoritmo Kamata
    double calculate_kamata(py::array_t<uint8_t> mask1, py::array_t<uint8_t> mask2, int lambda = 3) {
        auto points1 = mask_to_points(mask1);
        auto points2 = mask_to_points(mask2);
        
        if (points1.empty() || points2.empty()) {
            return 0.0;
        }
        
        return hausKamata(points1, points2, lambda);
    }
    
    // Algoritmo TAHA (necesitamos implementar hausdorffDistTaha2)
    double calculate_taha(py::array_t<uint8_t> mask1, py::array_t<uint8_t> mask2) {
        // Implementación similar usando el algoritmo TAHA
        // Nota: necesitarás exportar la función hausdorffDistTaha2 del proyecto original
        auto points1 = mask_to_points(mask1);
        auto points2 = mask_to_points(mask2);
        
        if (points1.empty() || points2.empty()) {
            return 0.0;
        }
        
        // Por ahora, usaremos una implementación básica
        // Deberás implementar la función hausdorffDistTaha2 y exportarla
        return calculate_basic_hausdorff(points1, points2);
    }
    
private:
    int max_level_;
    
    // Implementación básica de Hausdorff para TAHA
    double calculate_basic_hausdorff(const std::vector<Point>& set1, const std::vector<Point>& set2) {
        double max_dist_1_to_2 = 0.0;
        double max_dist_2_to_1 = 0.0;
        
        // Calcular h(set1, set2)
        for (const auto& p1 : set1) {
            double min_dist = std::numeric_limits<double>::max();
            for (const auto& p2 : set2) {
                double dist = std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + 
                                       (p1.y - p2.y) * (p1.y - p2.y));
                min_dist = std::min(min_dist, dist);
            }
            max_dist_1_to_2 = std::max(max_dist_1_to_2, min_dist);
        }
        
        // Calcular h(set2, set1)
        for (const auto& p2 : set2) {
            double min_dist = std::numeric_limits<double>::max();
            for (const auto& p1 : set1) {
                double dist = std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + 
                                       (p1.y - p2.y) * (p1.y - p2.y));
                min_dist = std::min(min_dist, dist);
            }
            max_dist_2_to_1 = std::max(max_dist_2_to_1, min_dist);
        }
        
        return std::max(max_dist_1_to_2, max_dist_2_to_1);
    }
};

// Funciones convenientes para usar directamente
double hausdorff_k2t_maxheap(py::array_t<uint8_t> mask1, py::array_t<uint8_t> mask2, int max_level = 10) {
    HausdorffCalculator calc(max_level);
    return calc.calculate_k2t_maxheap(mask1, mask2);
}

double hausdorff_k2t_maxheap_v2(py::array_t<uint8_t> mask1, py::array_t<uint8_t> mask2, int max_level = 10) {
    HausdorffCalculator calc(max_level);
    return calc.calculate_k2t_maxheap_v2(mask1, mask2);
}

double hausdorff_kamata(py::array_t<uint8_t> mask1, py::array_t<uint8_t> mask2, int lambda = 3, int max_level = 10) {
    HausdorffCalculator calc(max_level);
    return calc.calculate_kamata(mask1, mask2, lambda);
}
 
double hausdorff_taha(py::array_t<uint8_t> mask1, py::array_t<uint8_t> mask2, int max_level = 10) {
    HausdorffCalculator calc(max_level);
    return calc.calculate_taha(mask1, mask2);
}
//ELABORACION PROPIA
PYBIND11_MODULE(hausdorff_k2tree_core, m) { 
    m.doc() = "Hausdorff K2Tree algorithms for Python";
    
    // Clase principal
    py::class_<HausdorffCalculator>(m, "HausdorffCalculator")
        .def(py::init<int>(), py::arg("max_level") = 10)
        .def("calculate_k2t_maxheap", &HausdorffCalculator::calculate_k2t_maxheap,
             "Calculate Hausdorff distance using K2T-MAXHEAP algorithm",
             py::arg("mask1"), py::arg("mask2"))
        .def("calculate_k2t_maxheap_v2", &HausdorffCalculator::calculate_k2t_maxheap_v2,
             "Calculate Hausdorff distance using K2T-MAXHEAPv2 algorithm",
             py::arg("mask1"), py::arg("mask2"))
        .def("calculate_kamata", &HausdorffCalculator::calculate_kamata,
             "Calculate Hausdorff distance using Kamata algorithm",
             py::arg("mask1"), py::arg("mask2"), py::arg("lambda") = 3)
        .def("calculate_taha", &HausdorffCalculator::calculate_taha,
             "Calculate Hausdorff distance using TAHA algorithm",
             py::arg("mask1"), py::arg("mask2"));
    
    // Funciones convenientes
    m.def("hausdorff_k2t_maxheap", &hausdorff_k2t_maxheap,
          "Calculate Hausdorff distance using K2T-MAXHEAP algorithm",
          py::arg("mask1"), py::arg("mask2"), py::arg("max_level") = 10);
    
    m.def("hausdorff_k2t_maxheap_v2", &hausdorff_k2t_maxheap_v2,
          "Calculate Hausdorff distance using K2T-MAXHEAPv2 algorithm",
          py::arg("mask1"), py::arg("mask2"), py::arg("max_level") = 10);
    
    m.def("hausdorff_kamata", &hausdorff_kamata,
          "Calculate Hausdorff distance using Kamata algorithm",
          py::arg("mask1"), py::arg("mask2"), py::arg("lambda") = 3, py::arg("max_level") = 10);
    
    m.def("hausdorff_taha", &hausdorff_taha,
          "Calculate Hausdorff distance using TAHA algorithm",
          py::arg("mask1"), py::arg("mask2"), py::arg("max_level") = 10);
}
