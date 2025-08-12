# 🏎️ Agente de Conducción Autónoma con Aprendizaje por Refuerzo Profundo

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Horas Entrenamiento](https://img.shields.io/badge/Horas%20Entrenamiento-220+-green.svg)]()
[![Episodios](https://img.shields.io/badge/Episodios-5600+-brightgreen.svg)]()

> **Implementación de Deep Reinforcement Learning para navegación autónoma de vehículos en entornos dinámicos de carreras**

📖 **[English Version](README.md)** | **[Versión en Español](README_ES.md)**

### 🌍 Contexto del Proyecto
Este proyecto fue desarrollado como **Trabajo Final de Grado** en la **Universitat Oberta de Catalunya (UOC), España**. La implementación y el código están escritos en español, mientras que la documentación se proporciona en inglés para mayor accesibilidad internacional y presentación profesional.

## 🚀 Resumen del Proyecto

Este proyecto implementa un **agente de Optimización de Política Proximal (PPO)** entrenado desde cero para navegar circuitos de carreras de forma autónoma en Trackmania 2020. El agente aprende a tomar decisiones de conducción en tiempo real utilizando datos de sensores LIDAR, logrando mejoras significativas de rendimiento a través del aprendizaje por refuerzo profundo.

### 🎯 Logros Principales
- **📈 625% de mejora en rendimiento** a lo largo de 5,600+ episodios de entrenamiento
- **🧠 Implementación personalizada de PPO** construida desde cero
- **📊 Optimización avanzada de políticas** usando arquitectura actor-crítico
- **🏁 Navegación autónoma** en entornos complejos de carreras
- **🔬 Metodología experimental rigurosa** con validación estadística

## 🛠️ Arquitectura Técnica

### Diseño del Espacio de Estados (83 dimensiones)
- **Sensores LIDAR**: 4 marcos temporales × 19 rayos = 76 mediciones de distancia
- **Dinámica del Vehículo**: 1 característica de velocidad
- **Historial de Acciones**: 6 características (2 acciones previas × 3 dimensiones)

### Espacio de Acciones (Control Continuo)
- **Aceleración/Frenado**: [-1, 1] (acelerador adelante/atrás)
- **Control de Dirección**: [-1, 1] (ángulo de ruedas izquierda/derecha)
- **Control Auxiliar**: [-1, 1] (control adicional del vehículo)

## 📊 Métricas de Rendimiento

| Fase de Entrenamiento | Recompensa Promedio | Longitud Episodio | Tasa de Éxito | Mejoras Clave |
|----------------------|--------------------|--------------------|---------------|---------------|
| **Inicial** | 12,23 | ~500 pasos | <15% | Exploración aleatoria |
| **Medio Entrenamiento** | 17,83 | ~1000 pasos | ~40% | Navegación básica |
| **Final** | 87,17 | 2500 pasos | >85% | **Conducción experta** |
| **Mejora Total** | **+625%** | **+400%** | **+467%** | **Política convergida** |

### Estadísticas de Entrenamiento
- **Episodios Totales**: 5,608
- **Pasos de Tiempo Totales**: 4,529,231
- **Duración del Entrenamiento**: 220 horas, 1 minuto, 22 segundos
- **Rendimiento Máximo**: 111,35 de recompensa (Episodio 5503)
- **Convergencia**: Rendimiento estable en los últimos 200 episodios

### Validación Académica
**Trabajo Final de Grado**: "Entrenamiento de un agente de aprendizaje por refuerzo en Trackmania"
- **Institución**: Universitat Oberta de Catalunya (UOC)
- **Año**: 2025
- **Programa**: Grado en Ingeniería Informática - Especialización en Inteligencia Artificial
- **Tutor**: Gabriel Moyà Alcover
- **Calificación**: [9,3]

### Relevancia Industrial
Este proyecto demuestra habilidades directamente aplicables a:
- **Vehículos Autónomos**: Toma de decisiones en tiempo real y planificación de rutas
- **Robótica**: Control continuo e integración de sensores
- **IA de Juegos**: Comportamiento de agente inteligente y sistemas en tiempo real
- **Investigación en Machine Learning**: Implementación avanzada de algoritmos RL

## 📚 Tecnologías y Habilidades Demostradas

### Tecnologías Principales de IA/ML
- **Aprendizaje por Refuerzo Profundo** (PPO, Actor-Crítico, Gradientes de Política)
- **Diseño de Arquitectura de Redes Neuronales** (PyTorch, Implementaciones Personalizadas)
- **Optimización de Hiperparámetros** (Ajuste sistemático, Análisis de rendimiento)
- **Visión Computacional** (Procesamiento LIDAR, Representación de estado)

### Excelencia en Ingeniería de Software
- **Programación Orientada a Objetos** (Arquitectura limpia, Patrones de diseño)
- **Optimización de Rendimiento** (Aceleración GPU, Algoritmos eficientes)
- **Sistemas en Tiempo Real** (Toma de decisiones de baja latencia, Integración de sistemas)

### Habilidades de Investigación y Análisis
- **Diseño Experimental** (Pruebas de hipótesis, Validación estadística)
- **Análisis y Visualización de Datos** (Matplotlib, Métricas estadísticas)
- **Escritura Técnica** (Trabajo Final de Grado, Documentación clara)
- **Resolución de Problemas** (Depuración de algoritmos, Optimización de rendimiento)

## 📖 Referencias y Recursos

### Artículos Académicos Clave
- Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- Haarnoja et al. (2018) - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- Sutton & Barto (2018) - "Reinforcement Learning: An Introduction"
- Lillicrap et al. (2015) - "Continuous Control with Deep Reinforcement Learning"

### Recursos Técnicos
- **Framework TMRL**: Proyecto comunitario Trackmania Reinforcement Learning
- **OpenPlanet**: Middleware de integración con Trackmania
- **Documentación PyTorch**: Referencia del framework de deep learning
- **Gymnasium**: Estándar de entornos de aprendizaje por refuerzo

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles completos.

La Licencia MIT permite uso tanto académico como comercial, modificación y distribución.

## 👨‍💻 Autor

**Carlos Expósito Carrera**
- 🎓 Graduado en Ingeniería Informática, Especialización en Inteligencia Artificial
- 🏫 Universitat Oberta de Catalunya (UOC)
- 🔬 Enfoque de Investigación: Deep Reinforcement Learning, Sistemas Autónomos
- 📧 Email: [cexposito1@gmail.com](mailto:cexposito1@gmail.com)
- 💼 LinkedIn: [Carlos Expósito Carrera](https://www.linkedin.com/in/carlos-exposito-carrera/)
- 🐙 GitHub: [@moosemaniac](https://github.com/moosemaniac)