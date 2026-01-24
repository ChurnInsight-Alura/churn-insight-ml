# **Fintech - Customer Churn**

<img width="1024" height="638" alt="ChatGPT Image Dec 19, 2025, 10_52_55 AM" src="https://github.com/user-attachments/assets/c6570e83-b0dd-474e-86ba-ebe7fee664ac" />


---

## Índice 📋

1. Descripción del proyecto
2. Acceso al proyecto
3. Etapas del proyecto
4. Catálogo de datos
5. Resultados y conclusiones
6. Tecnologías utilizadas
7. Agradecimientos
8. Desarrolladores del proyecto

---


## 1. Descripción del proyecto 📚
---
### **Contexto del Negocio**
En el competitivo sector Fintech, el costo de adquisición de nuevos clientes (CAC) es significativamente más alto que el costo de retención de los existentes. **CusTech**, una empresa ficticia del sector financiero, ha detectado una tasa de abandono (churn) que amenaza la estabilidad de su cartera. La capacidad de predecir qué clientes tienen un alto riesgo de cancelar sus productos permite a la empresa tomar medidas proactivas, optimizando recursos y enfocando estrategias de fidelización.

### **Objetivo**
Reducir la pérdida de cartera vigente y aumentar el Lifetime Value (LTV) del cliente mediante la implementación de un modelo de Machine Learning capaz de predecir la probabilidad de abandono con antelación, permitiendo intervenciones estratégicas antes de que el cliente finalice su relación con el banco.

### **La Solución**
Se desarrolló un pipeline integral que culmina en una **FastAPI** desplegable. La solución incluye:
1.  **Predicción de Churn:** Utilizando un modelo **LightGBM** optimizado.
2.  **Segmentación de Clientes:** Agrupamiento mediante **K-Means** para entender perfiles.
3.  **Matriz de Prioridad:** Clasificación automática de **Prioridad de Intervención** según el segmento y la probabilidad de abandono.

## 2. Acceso al proyecto 📂
---
Para obtener el proyecto tienes dos opciones:

1. Clonar el repositorio utilizando la línea de comandos. Solo debes dirigirte al directorio donde deseas clonar el mismo e ingresar el comando:<br><br>
   `git clone https://github.com/ChurnInsight-Alura/churn-insight-ml`

2. O puedes descargarlo directamente desde el repositorio en GitHub en el siguiente enlace:<br>

   [https://github.com/ChurnInsight-Alura/churn-insight-ml](https://github.com/ChurnInsight-Alura/churn-insight-ml)

   Esto te llevará a la siguiente pantalla, donde deberás seguir los siguientes pasos:

<img width="1736" height="929" alt="image" src="https://github.com/user-attachments/assets/24daca25-8f42-4ae4-8bce-3d8453c76fce" />
   
Esto descargará un archivo comprimido `.zip`, que podrás alojar en el directorio que desees.

### **2.1 Ejecución de la API con Docker (Recomendado)**

Si dispones de Docker instalado, puedes levantar la API completa (incluyendo dependencias y modelos) con un solo comando.

1. Navega a la carpeta raíz del proyecto clonado y luego a la carpeta llamada 📂api.
2. Construye y levanta el contenedor:

```bash
docker build -t custech-api .
docker run -d -p 8000:8000 custech-api
```

### **2.2 Pruebas de la API**

Una vez que el contenedor esté corriendo (o si ejecutaste uvicorn main:app --reload localmente), puedes acceder a la documentación interactiva generada automáticamente por Swagger UI.

1. Abre tu navegador y ve a: http://localhost:8000/docs
2. Verás los endpoints disponibles para realizar predicciones individuales o por lotes.

Endpoints principales:

* POST /predict/user: Predicción para un solo cliente (JSON).
* POST /predict/batch: Predicción masiva para un archivo con lista de JSONs.
* POST /predict/batch_stats: Predicción masiva para un archivo con lista de JSONs que genera estadísticas generales para dashboard administrativo (gerencia).

## 3. Etapas del proyecto 📝
---

1. [Consolidación de datasets](https://github.com/ChurnInsight-Alura/churn-insight-ml/blob/main/datasets_consolidation_CusTech.ipynb)
2. [Análisis Exploratorio de Datos (EDA)](https://github.com/ChurnInsight-Alura/churn-insight-ml/blob/main/EDA_CusTech_Ignacio.ipynb)
3. [Modelado de datos](https://github.com/ChurnInsight-Alura/churn-insight-ml/blob/main/Modeling_CusTech.ipynb)
4. [FastAPI](https://github.com/ChurnInsight-Alura/churn-insight-ml/tree/main/api)
5. [Generación de datos para DEMO](https://github.com/ChurnInsight-Alura/churn-insight-ml/blob/main/datos_demo_CusTech.ipynb)



## 4. Catálogo de Datos
---

#### **Ventanas Temporales**

Para llevar los datos de tiempo a un formato tabular se consideró una ventana de análisis de 365 días.

| VENTANA       | Período          | Comienzo          | Final            | Descripción                                           |
|---------------|------------------|-------------------|------------------|-------------------------------------------------------|
| WINDOW_1      | Q1               | 2024-12-31        | 2025-04-04       | Primer trimestre para observación del comportamiento  | 
| WINDOW_2      | Q2               | 2025-04-05        | 2025-07-03       | Segundo trimestre para observación del comportamiento |
| WINDOW_3      | Q3               | 2025-07-04        | 2025-10-01       | Tercer trimestre para observación del comportamiento  |
| CUTOFF_DATE   | Q4               | 2025-10-02        | 2025-12-31       | Ventana de Churn                                      |

### **Clientes**

| Variable	        | Tipo de dato      | Definición Funcional	                                      |
|-------------------|-------------------|------------------------------------------------------------|
| `RowNumber`	     | String            | Índice de fila                                             | 
| `CustomerId`	     | Integer           | Identificador único del cliente.	                          | 
| `Surname`	        | String            | Apellido del cliente.	                                   |
| `CreditScore`	  | Integer           | Puntaje crediticio                                         |
| `Geography`	     | String            | País de residencia                                         | 
| `Gender`	        | String            | Género.                                                    |
| `Age`	           | Integer           | Edad.                                                      |
| `Tenure`	        | Integer           | Antigüedad en años                                         |
| `Balance`         | Float             | Balance actual en su cuenta                                |
| `NumOfProducts`   | Integer           | Cantidad de productos contratados                          | 
| `HasCrCard`       | Integer           | Tiene tarjeta de crédito, si o no (1 o 0)                  | 
| `IsActiveMember`  | Integer           | Utiliza servicios y promociones del banco, si o no (1 o 0) |
| `EstimatedSalary` | Float             | Salario mensual estimado                                   |


### **Transacciones**

| Variable	          | Tipo de dato   | Definición Funcional                                        |
|---------------------|----------------|-------------------------------------------------------------|
| `TransactionId`     | String	      | Ingreso mensual del cliente.	                             |
| `CusomerId`	       | Int            | Identificador único del cliente que realizó la transacción  |
| `TransactionDate`	 | Datetime       | Renta declarada en la solicitud.	                          |
| `Amount`	          | Float          | Monto de la transacción                                     |
| `TransactionType`	 | String         | Tipo de transacción realizada (Ej.: TRANSFER, PAYMENT, etc) |

#### Feature Engineering

| Feature                   | Tipo                | Descripción                                                                          | 
|---------------------------|---------------------|--------------------------------------------------------------------------------------| 
| `avg_tx_amount`           | Float               | Monto promedio de las transacciones del cliente                                      |
| `days_since_last_tx`      | Int                 | Días desde que el cliente realizó la última transacción                              |
| `tx_q1q2_rate_of_change`  | Float               | Tasa de cambio en la cantidad de transacciones entre el primer y segundo trimestre   |
| `tx_q2q3_rate_of_change`  | Float               | Tasa de cambio en la cantidad de transacciones entre el segundo y tercer trimestre   |


**Notas**: 

* `client` representa una fila
* `df_tx` representa el conjunto de transacciones del cliente

> **`avg_tx_amount`**
```Python
client['avg_tx_amount'] = df_tx.groupby('CustomerId')['Amount'].mean()
```

> **`days_since_last_tx`**
```Python
client['days_since_last_tx'] = (CUTOFF_DATE - df_tx['TransactionDate'].max()).dt.days
```

> **`tx_q1q2_rate_of_change`**
```Python
client['tx_q1q2_rate_of_change'] = (client['total_tx_q2'] - client['total_tx_q1']) / client['total_tx_q1']
```

> **`tx_q2q3_rate_of_change`**
```Python
client['tx_q2q3_rate_of_change'] = (client['total_tx_q3'] - client['total_tx_q2']) / client['total_tx_q2']
```


### **Interacciones con la Aplicación**

| Variable	     | Tipo de dato   | Definición Funcional                                  |
|----------------|----------------|-------------------------------------------------------|
| `SessionId`    | String         | Identificador único de log de sesión                  |
| `CustomerId`   | Int            | Identificador único del cliente que ejecutó la sesión |
| `SessionDate`  | Datetime       | Fecha de la sesión                                    |
| `DurationMin`  | Float          | Duración de la sesion en minutos                      |
| `UsedTransfer` | Int            | Usó opción de transferencia, si o no (1 o 0)          |
| `UsedPayment`  | Int            | Usó opción de pago, si o no (1 o 0)                   |
| `UsedInvest`   | Int            | Usó opción de inversión, si o no (1 o 0)              |
| `OpenedPush`   | Int            | Abrió notificación, si o no (1 o 0)                   |
| `FailedLogin`  | Int            | Falló el inicio de sesión, si o no (1 o 0)            |

#### Feature Engineering

| Feature                   | Tipo                | Descripción                                                                          | 
|---------------------------|---------------------|--------------------------------------------------------------------------------------| 
| `avg_ss_duration`         | Float               | Duración promedio de las sesiones del cliente                                        |
| `std_ss_duration`         | Float               | Desviación estándar de la duración de las sesiones del cliente                       |
| `days_since_last_ss`      | Int                 | Días desde que el cliente realizó la última transacción                              |
| `ss_q1q2_rate_of_change`  | Float               | Tasa de cambio en la cantidad de sesiones entre el primer y segundo trimestre        |
| `ss_q2q3_rate_of_change`  | Float               | Tasa de cambio en la cantidad de sesiones entre el segundo y tercer trimestre        |
| `failed_ratio_spike_q2`   | Float               | Diferencia entre el ratio del primer trimestre y el segundo trimestre                |
| `failed_ratio_spike_q3`   | Float               | Diferencia entre el ratio del primer trimestre y el segundo trimestre                |
| `failed_ratio_volatility` | Float               | Desviación estandar calculada a partir del ratio de fallos de los 3 tirmestres       |

Explicación del feature engineering para variables compuestas:

**Notas**: 

* `client` representa una fila
* `df_ss` representa el conjunto de transacciones del cliente

> **`avg_ss_duration`**
```Python
client['avg_ss_duration'] = df_ss.groupby('CustomerId')['DurationMin'].mean()
```

> **`std_ss_duration`**
```Python
client['avg_ss_duration'] = df_ss.groupby('CustomerId')['DurationMin'].std()
```

> **`days_since_last_ss`**
```Python
client['days_since_last_ss'] = (CUTOFF_DATE - df_ss['SessionDate'].max()).dt.days
```

> **`ss_q1q2_rate_of_change`**
```Python
client['ss_q1q2_rate_of_change'] = (client['total_ss_q2'] - client['total_ss_q1']) / client['total_ss_q1']
```

> **`ss_q2q3_rate_of_change`**
```Python
client['ss_q2q3_rate_of_change'] = (client['total_tx_q3'] - client['total_tx_q2']) / client['total_tx_q2']
```

> **`failed_ratio_spike_q2`**
```Python
client['failed_ratio_spike_q2'] = (client['total_failed_ss_q2'] / client['total_ss_q2']) - (client['total_failed_ss_q1'] / client['total_ss_q1'])
```

> **`failed_ratio_spike_q3`**
```Python
client['failed_ratio_spike_q3'] = (client['total_failed_ss_q3'] / client['total_ss_q3']) - (client['total_failed_ss_q2'] / client['total_ss_q2'])
```

> **`failed_ratio_volatility`** 
```Python
client['failed_ratio_volatility'] = [(client['total_failed_ss_q1'] / client['total_ss_q1']), (client['total_failed_ss_q2'] / client['total_ss_q2']), (client['total_failed_ss_q3'] / client['total_ss_q3'])].std()
```



#### **Target**

| Variable      | Tipo de dato   | Descripción Funcional                                                                      | 
|---------------|----------------|--------------------------------------------------------------------------------------------| 
| `Exited`      | Int            | Condición de abando: 1 = Churn (Canceló) | 0 = No Churn (Permanencia)                      |



## 5. Resultados y conclusiones
---

### **5.1 Comparativa de Modelos (Umbral por defecto 0.5)**
Para la selección del modelo final, se realizaron aproximadamente 4 experimentos por familia de algoritmos, seleccionando un representante de cada una para una evaluación comparativa profunda.

| Model               | Stage    | Accuracy   | Precision  | Recall    | F1-score  | AUC        | Umbral   |
|---------------------|----------|------------|------------|-----------|-----------|------------|----------|
| Random Forest       | Test     | 0.8786     | 0.6864     | 0.6658    | 0.6759    | 0.8999     | 0.5      |
| Logistic Regression | Test     | 0.7780     | 0.4524     | **0.7945**| 0.5765    | 0.8663     | 0.5      |
| LightGBM Classifier | Test     | 0.8828     | 0.6872     | 0.7041    | **0.6955**| **0.9040** | 0.5      |
| XGBoost Classifier  | Test     | 0.8791     | 0.6895     | 0.6630    | 0.6760    | 0.8919     | 0.5      |


Al observar las métricas generales, LightGBM Classifier muestra el mejor desempeño integral (Mejor F1-Score y AUC). Aunque la Regresión Logística presenta el Recall más alto, su Precisión es extremadamente baja (0.45), lo que implica un exceso de falsos positivos.

### **5.2 Estrategia de Umbrales de Decisión**
El objetivo de negocio inicial fue capturar el 80% de los fugados (Recall = 0.8). Sin embargo, al ajustar los umbrales para lograr esta sensibilidad, la Precisión de todos los modelos cayó por debajo de 0.60, volviendo las campañas de retención costosas e ineficientes.

Se optó por un compromiso estratégico de Recall = 0.75, logrando identificar 3 de cada 4 fugas potenciales manteniendo una precisión aceptable.

| Model               | Accuracy | Precision | Recall | F1-score | AUC    | Umbral Ajustado |
|---------------------|----------|-----------|--------|----------|--------|-----------------|
| Random Forest       | 0.8598   | 0.6048    | 0.7589 | 0.6731   | 0.8999 | 0.40            |
| Logistic Regression | 0.8150   | 0.5094    | 0.7425 | 0.6042   | 0.8663 | 0.56            |
| LightGBM Classifier | 0.8583   | 0.6009    | 0.7589 | 0.6707   | 0.9040 | 0.39            |
| XGBoost Classifier  | 0.8557   | 0.5944    | 0.7589 | 0.6667   | 0.8919 | 0.27            |

### **5.3 Análisis de Generalización (Overfitting vs Underfitting)**

El análisis de las brechas de rendimiento entre entrenamiento y prueba reveló comportamientos críticos:

* **Sobreajuste Severo (Random Forest y XGBoost):** Ambos modelos mostraron métricas casi perfectas en entrenamiento (Recall ~99%) pero caídas drásticas en prueba (~32% de pérdida). "Memorizan" el set de entrenamiento pero fallan al generalizar.
* **Sobreajuste Moderado (LightGBM):** Si bien presenta una caída en rendimiento, es estructuralmente más robusto, manteniendo un Recall y F1-Score en test superiores a sus competidores de árboles.
* **Subajuste (Regresión Logística):** Muestra gran estabilidad, pero su simplicidad le impide capturar la complejidad no lineal de los datos.

### **5.4 Modelo Campeón: LightGBM Classifier** 🏆

Se seleccionó LightGBM como el modelo final para producción debido a que:

* Ofrece el mejor equilibrio entre AUC (0.9040) y capacidad de generalización.
* Logra el objetivo de Recall (0.75) con una precisión competitiva.
* Mitiga el overfitting severo observado en XGBoost y Random Forest.

### **5.5 Interpretabilidad: Factores de Influencia (SHAP)**

A través del análisis de SHAP Values, desglosamos la "Caja Negra" del modelo para entender qué variables pesan más en la decisión de abandono:

> **`Age`:** Es el mayor predictor individual. Se puede observar que valores elevados en dicha variable conllevan un fuerte SHAP positivo, los cuales contribuyen a la evasión. Existe una gran concentración de clientes que abandonan entre 38 y 51 años. 

> **`NumOfProducts`:** Si bien valores elevados para este feautre parece actuar como protector contra el abandono, también se observa que valores altos pero no máximos contribuyen al abandono, esto podría indicar insatisfacción con algún producto en particular.

> **`ss_q2q3_rate_of_change`:** Fuerte predictor de Churn. Si este ratio tiene un valor negativo (estandarizado) entre -1.5 y -0.5, las probabilidades relativas de abandono aumentan considerablemente -> **-0.32 UMBRAL CRÍTICO DE MONITOREO**

> **`IsActiveMember`:** Como es de esperarse, que un cliente esté clasificado como "Activo", contribuye a la retención, mientras aquellos que no interactúan con la empresa tienen valores SHAP positivos.

> **`days_since_last_tx`:** Factor de alto riesgo, a medida que pasan los días sin que el cliente realize transacciones, más aumentan las probabilidades relativas de abandono.

> **`Gender_Male`:** El hecho de ser mujer (`Gender_Male = 0`) actúa como potenciador en las probabilidades relativas de Churn. Este es un punto crítico a investigar dado que puede haber sesgos de género en productos, ofertas y/o condiciones -> **IMPORTANTE REVISAR POLÍTICAS DE EMPRESA**.

> **`Geography_Germany`:** Ser alemán aumenta las probabilidades relativas de abandono. Será necesario investigar las razones por las cuales los clientes de este país son más propensos a abandonar la empresa.

> **`days_since_last_ss`:** Al igual que días desde la última transacción, mientras más días pasa un cliente sin conectarse a la aplicación de la empresa, más aumenta el riesgo de abandono.

> **`tx_q2q3_rate_of_change`:** Valores altos contribuyen a la retención, mientras que valores bajos están fuertemente asociados al abandono. Una dimsinución en la cantidad de transacciones realizadas en el último período (este feature evalúa la tasa de cambio entre en segundo y tercer trimestre) es un fuerte indicador de abandono.  **-0.03 UMBRAL CRÍTICO DE MONITOREO**

> **`Balance`:** **CRÍTICO** -> Tener un alto balance contribuye a las probabilidades relativas de abandono, esto refleja que los clientes que deciden dejar la empresa son de alto valor. Resulta de suma importancia investigar este fenómeno en profundidad.


## 6. Tecnologías utilizadas 🛠️

* `Python` (3.10+)
* `Jupyter`
* `FastAPI`
* `Docker`
* `Git y GitHub`


## 7. Agradecimientos 🤝

Este proyecto fue posible gracias al apoyo y formación brindada por:

* **Oracle Next Education (ONE):** Por proporcionar la base educativa y el desafío técnico.
* **Alura Latam:** Por la excelencia en los cursos de especialización en Data Science.
* **NoCountry:** Por facilitar el entorno de simulación laboral que permitió llevar este proyecto a un nivel profesional.

Un agradecimiento especial a mis **compañeros de equipo** por la colaboración durante las fases de testing e integración y su **excelente desarrollo tanto de back-end como front-end** para la presentación de una **solución funcional, atractiva y con impacto de negocio**.


## 8. Desarrollador del proyecto 👷

* **Ignacio Majo**
  - Rol: Data Scientist - ML Engineer
