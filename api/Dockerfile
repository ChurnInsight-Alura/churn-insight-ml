# 1. Imagen base de Python oficial (ligera)
FROM python:3.12-slim

# 2. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalación de dependencias del sistema
# LightGBM necesita libgomp1 para funcionar en Linux
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar requirements para aprovechar la caché de Docker
COPY requirements.txt .

# 5. Instalar librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar el resto del código (main.py y la carpeta model/)
COPY . .

# 7. Exponer en puerto 8000
EXPOSE 8000

# 8. Comando para arrancar la API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]