/*
 * main.cc - Versión paralela del análisis forense de imágenes
 *
 * Estrategia de paralelización:
 *  - Paralelismo por tareas (std::async): ejecución simultánea de procesos independientes
 *  - Paralelismo de datos (OpenMP): distribución del trabajo interno en bucles
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "png.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include "utils/image.h"
#include "utils/dct.h"
#include <string>
#include <chrono>
#include <future>
#include <omp.h>

// ============================================================
// Kernels SRM
// Definen máscaras para detectar patrones de ruido en la imagen
// ============================================================
Image<float> get_srm_3x3() {
    Image<float> kernel(3, 3, 1);
    kernel.set(0,0,0,-1); kernel.set(0,1,0, 2); kernel.set(0,2,0,-1);
    kernel.set(1,0,0, 2); kernel.set(1,1,0,-4); kernel.set(1,2,0, 2);
    kernel.set(2,0,0,-1); kernel.set(2,1,0, 2); kernel.set(2,2,0,-1);
    return kernel;
}

Image<float> get_srm_5x5() {
    Image<float> kernel(5, 5, 1);
    kernel.set(0,0,0,-1); kernel.set(0,1,0, 2); kernel.set(0,2,0,-2); kernel.set(0,3,0, 2); kernel.set(0,4,0,-1);
    kernel.set(1,0,0, 2); kernel.set(1,1,0,-6); kernel.set(1,2,0, 8); kernel.set(1,3,0,-6); kernel.set(1,4,0, 2);
    kernel.set(2,0,0,-2); kernel.set(2,1,0, 8); kernel.set(2,2,0,-12);kernel.set(2,3,0, 8); kernel.set(2,4,0,-2);
    kernel.set(3,0,0, 2); kernel.set(3,1,0,-6); kernel.set(3,2,0, 8); kernel.set(3,3,0,-6); kernel.set(3,4,0, 2);
    kernel.set(4,0,0,-1); kernel.set(4,1,0, 2); kernel.set(4,2,0,-2); kernel.set(4,3,0, 2); kernel.set(4,4,0,-1);
    return kernel;
}

Image<float> get_srm_kernel(int size) {
    assert(size == 3 || size == 5);
    return (size == 3) ? get_srm_3x3() : get_srm_5x5();
}

// ============================================================
// compute_srm
// Aplica convolución con el kernel SRM.
// Paralelismo: OpenMP distribuye filas de la imagen entre hilos
// ============================================================
Image<unsigned char> compute_srm(const Image<unsigned char> &image, int kernel_size) {
    auto begin = std::chrono::steady_clock::now();
    std::cout << "[SRM " << kernel_size << "x" << kernel_size << "] Iniciando..." << std::endl;

    Image<float> gray = image.to_grayscale().convert<float>();
    Image<float> kernel = get_srm_kernel(kernel_size);

    int h = gray.height, w = gray.width;
    int ks = kernel_size;
    Image<float> srm(w, h, 1);

    // Convolución: cada hilo procesa filas independientes
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            float sum = 0.0f;
            for (int u = 0; u < ks; u++) {
                for (int v = 0; v < ks; v++) {
                    int s = j + u - ks / 2;
                    int t = i + v - ks / 2;
                    if (s < 0 || s >= h || t < 0 || t >= w) continue;
                    sum += gray.get(s, t, 0) * kernel.get(u, v, 0);
                }
            }
            srm.set(j, i, 0, sum / (ks * ks));
        }
    }

    // Valor absoluto y cálculo de min/max en paralelo
    float max_val = -1e9f, min_val = 1e9f;
    #pragma omp parallel for reduction(max:max_val) reduction(min:min_val)
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            float v = std::abs(srm.get(j, i, 0));
            srm.set(j, i, 0, v);
            if (v > max_val) max_val = v;
            if (v < min_val) min_val = v;
        }
    }

    // Normalización a rango [0,255]
    Image<unsigned char> result(w, h, 1);
    float range = (max_val - min_val > 0) ? (max_val - min_val) : 1.0f;

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            float normalized = (srm.get(j, i, 0) - min_val) / range * 255.0f;
            result.set(j, i, 0, (unsigned char)normalized);
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "[SRM " << kernel_size << "x" << kernel_size << "] Tiempo: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms (hilos: " << omp_get_max_threads() << ")" << std::endl;
    return result;
}

// ============================================================
// compute_dct
// Divide la imagen en bloques independientes y aplica DCT.
// Paralelismo: cada bloque se procesa de forma independiente
// ============================================================
Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert) {
    auto begin = std::chrono::steady_clock::now();
    std::cout << "[DCT" << (invert ? " inv" : "") << " " << block_size << "x" << block_size << "] Iniciando..." << std::endl;

    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);
    int n_blocks = (int)blocks.size();

    // Paralelismo de datos sobre bloques
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_blocks; i++) {
        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, blocks[i], 0);

        if (invert) {
            for (int k = 0; k < blocks[i].size / 2; k++)
                for (int l = 0; l < blocks[i].size / 2; l++)
                    dctBlock[k][l] = 0.0f;

            dct::inverse(blocks[i], dctBlock, 0, 0.0, 255.);
        } else {
            dct::assign(dctBlock, blocks[i], 0);
        }

        dct::delete_matrix(dctBlock);
    }

    Image<unsigned char> result = grayscale.convert<unsigned char>();

    auto end = std::chrono::steady_clock::now();
    std::cout << "[DCT" << (invert ? " inv" : "") << "] Tiempo: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms (hilos: " << omp_get_max_threads() << ")" << std::endl;

    return result;
}

// ============================================================
// compute_ela
// Detecta diferencias entre imagen original y recomprimida.
// Limitado por operaciones de I/O.
// ============================================================
Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality) {
    auto begin = std::chrono::steady_clock::now();
    std::cout << "[ELA q=" << quality << "] Iniciando..." << std::endl;

    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp.jpg", grayscale, quality);
    Image<float> compressed = load_from_file("_temp.jpg").convert<float>();

    int h = grayscale.height, w = grayscale.width;
    Image<float> gray_f = grayscale.convert<float>();
    Image<float> diff(w, h, 1);

    // Diferencia por píxel
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++)
            diff.set(j, i, 0, std::abs(compressed.get(j, i, 0) - gray_f.get(j, i, 0)));

    float max_val = -1e9f, min_val = 1e9f;

    #pragma omp parallel for reduction(max:max_val) reduction(min:min_val)
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++) {
            float v = diff.get(j, i, 0);
            if (v > max_val) max_val = v;
            if (v < min_val) min_val = v;
        }

    float range = (max_val - min_val > 0) ? (max_val - min_val) : 1.0f;
    Image<unsigned char> result(w, h, 1);

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++)
            result.set(j, i, 0, (unsigned char)((diff.get(j, i, 0) - min_val) / range * 255.0f));

    auto end = std::chrono::steady_clock::now();
    std::cout << "[ELA] Tiempo: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;

    return result;
}

// ============================================================
// main
// Flujo:
//   1. Carga de imagen
//   2. Ejecución paralela de tareas independientes (std::async)
//   3. Sincronización y guardado de resultados
// ============================================================
int main(int argc, char **argv) {
    if (argc == 1) {
        std::cerr << "Uso: ./detect <imagen>" << std::endl;
        exit(1);
    }

    int block_size = 8;
    Image<unsigned char> image = load_from_file(argv[1]);

    auto total_start = omp_get_wtime();
    std::cout << "=== Análisis forense paralelo ===" << std::endl;
    std::cout << "Hilos OpenMP disponibles: " << omp_get_max_threads() << std::endl;

    // Lanzamiento de tareas independientes
    auto f_srm3 = std::async(std::launch::async, [&]() { return compute_srm(image, 3); });
    auto f_srm5 = std::async(std::launch::async, [&]() { return compute_srm(image, 5); });
    auto f_ela  = std::async(std::launch::async, [&]() { return compute_ela(image, 90); });
    auto f_dct_inv = std::async(std::launch::async, [&]() { return compute_dct(image, block_size, true); });
    auto f_dct_dir = std::async(std::launch::async, [&]() { return compute_dct(image, block_size, false); });

    // Sincronización y guardado
    save_to_file("srm_kernel_3x3.png", f_srm3.get());
    save_to_file("srm_kernel_5x5.png", f_srm5.get());
    save_to_file("ela.png",            f_ela.get());
    save_to_file("dct_invert.png",     f_dct_inv.get());
    save_to_file("dct_direct.png",     f_dct_dir.get());

    double total_end = omp_get_wtime();
    std::cout << "=== Tiempo total: "
              << (int)((total_end - total_start) * 1000)
              << " ms ===" << std::endl;

    return 0;
}