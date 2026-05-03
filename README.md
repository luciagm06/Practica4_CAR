# Práctica 4 — Computación de Alto Rendimiento

## Paralelismo a nivel de hilos: Paralelización mediante OpenMP y programación asíncrona del análisis forense de manipulación de imágenes digitales

**Lucía González Mandler**  
**Sofía Pérez Vásquez**  

**Universidad de Alicante**  
Curso 2025/2026

---

## Descripción

En esta práctica se ha desarrollado una versión paralela de un programa de análisis forense de imágenes digitales a partir de una implementación secuencial.

El objetivo principal ha sido reducir el tiempo de ejecución aprovechando mejor los recursos del procesador mediante técnicas de paralelismo a nivel de hilos.

Para ello se han utilizado:

- **OpenMP**, para paralelizar bucles internos.
- **std::async**, para ejecutar tareas independientes de forma concurrente.

---

## Procesos implementados

El programa realiza cinco análisis principales sobre una imagen de entrada:

- **SRM 3x3**
- **SRM 5x5**
- **ELA (Error Level Analysis)**
- **DCT directa**
- **DCT inversa**

Cada proceso genera una imagen resultado independiente.

---

## Estrategia de paralelización

Se ha seguido la siguiente estrategia:

### Paralelismo por tareas

Mediante `std::async` se ejecutan simultáneamente:

- SRM 3x3
- SRM 5x5
- ELA
- DCT directa
- DCT inversa

### Paralelismo de datos

Mediante **OpenMP** se reparten los cálculos internos entre varios hilos:

- Recorrido de píxeles en SRM
- Procesamiento de bloques en DCT
- Comparación y normalización en ELA

---

## Resultados obtenidos

| Hilos | Tiempo (ms) | Speedup |
|------|------------|---------|
| 1 | 2372 | 1.81 |
| 2 | 1275 | 3.36 |
| 4 | 795 | 5.39 |
| 8 | 538 | 7.97 |
| 16 | 500 | **8.57** |

La mejor ejecución se obtuvo con **16 hilos**, alcanzando una aceleración de **8.57x** respecto a la versión secuencial.

---

## Compilación

```bash
cmake -S . -B build
cmake --build build
```

---

## Ejecución

```bash
cd build
./detect <imagen.png>
OMP_NUM_THREADS=16 ./detect <imagen.png>
```
