# Text to Image Generation with MLX Stable Diffusion / Generación de Imágenes desde Texto con MLX Stable Diffusion

[🎥 Watch the Demo Video](https://www.loom.com/share/28f87c89e9c741068c69fdaefaa5e170)

[English](#english) | [Español](#español)

## English

### Description

This project implements a text-to-image generation system using Stable Diffusion optimized for Apple Silicon through MLX. It provides both a Jupyter notebook interface and a Gradio web UI for generating images from text descriptions.

### Features

- Optimized for Apple Silicon using MLX framework
- Uses Stable Diffusion 2.1 base model
- Includes model quantization for better performance
- Supports both simple and advanced generation parameters
- Interactive web interface with Gradio

### Requirements

- macOS with Apple Silicon
- Python 3.9+

### Installation

1. Clone the repository to your local machine
2. Install the required dependencies using pip and the requirements.txt file

### Usage

Option 1: Using Jupyter Notebook

1. Start Jupyter Notebook on your local machine
2. Open the text_to_image.ipynb notebook

Option 2: Direct Gradio Interface

1. Run `python app.py`

### Parameters

- **Prompt**: Text description of the desired image
- **Negative Prompt**: What you don't want in the image
- **Steps**: Number of denoising steps (higher = better quality, slower generation)
- **Guidance Scale**: How closely to follow the prompt (higher = more faithful, less creative)
- **Seed**: For reproducible results

---

## Español

### Descripción

Este proyecto implementa un sistema de generación de imágenes a partir de texto utilizando Stable Diffusion optimizado para Apple Silicon mediante MLX. Proporciona tanto una interfaz de Jupyter notebook como una interfaz web Gradio para generar imágenes a partir de descripciones textuales.

### Características

- Optimizado para Apple Silicon usando el framework MLX
- Utiliza el modelo base Stable Diffusion 2.1
- Incluye cuantización del modelo para mejor rendimiento
- Soporta parámetros de generación simples y avanzados
- Interfaz web interactiva con Gradio

### Requisitos

- macOS con Apple Silicon
- Python 3.9+

### Uso

Opción 1: Usando Jupyter Notebook

1.  Iniciar Jupyter Notebook en su máquina local
2.  Abrir el notebook text_to_image.ipynb

Opción 2: Interfaz Gradio Directa

1. Ejecutar `python app.py`

- ### Parámetros

* **Prompt**: Descripción textual de la imagen deseada
* **Prompt Negativo**: Lo que no deseas en la imagen
* **Pasos**: Número de pasos de eliminación de ruido (mayor = mejor calidad, generación más lenta)
* **Guidance Scale**: Qué tanto seguir el prompt (mayor = más fiel, menos creativo)
* **Seed**: Para resultados reproducibles
