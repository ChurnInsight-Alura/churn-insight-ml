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


### **Objetivo** 

Reducir la pérdida de cartera vigente y aumentar el Lifetime Value (LTV) del cliente mediante 

### **La Solución** 

Hemos desarrollado un MVP End-to-End que integra un modelo de Machine Learning con una API REST funcional.

***Enfoque de Data Science:*** Entrenamos un modelo de clasificación binaria utilizando datos históricos transaccionales y de interacción con la aplicación, así como datos demográficos. Identificamos variables clave <VARIABLES> para evitar la deserción.

***Enfoque de Backend:*** Disponibilizamos el modelo a través de una API (Java/Spring Boot) que permite al negocio consultar el riesgo de un cliente en tiempo real, devolviendo una predicción clara ("Deserción" / "No Deserción") y su probabilidad asociada.



## 2. Acceso al proyecto 📂
---
Para obtener el proyecto tienes dos opciones:

1. Clonar el repositorio utilizando la línea de comandos. Solo debes dirigirte al directorio donde deseas clonar el mismo e ingresar el comando:<br><br>
   `git clone https://github.com/<link_repositorio>`

2. O puedes descargarlo directamente desde el repositorio en GitHub en el siguiente enlace:<br>

   [https://github.com/<link_repositorio>](https://github.com/<link_repositorio>)

   Esto te llevará a la siguiente pantalla, donde deberás seguir los siguientes pasos:


   
Esto descargará un archivo comprimido `.zip`, que podrás alojar en el directorio que desees.


### **NOTA**:

***Aquí deberemos ver la manera de mostrar como "ensamblar" los 3 repositorios para que el usuario obtenga un producto funcional.***

## 3. Etapas del proyecto 📝
---
<br><br><br>
Aquí podemos generalizar inicialmente a cada repositorio: Back-end, Front-end, Data Science. Y luego dentro de cada "sección explicar las etapas realizadas para
cada área.

<br><br><br>

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
```
client['avg_tx_amount'] = df_tx.groupby('CustomerId')['Amount'].mean()
```

> **`days_since_last_tx`**
```
client['days_since_last_tx'] = (CUTOFF_DATE - df_tx['TransactionDate'].max()).dt.days
```

> **`tx_q1q2_rate_of_change`**
```
client['tx_q1q2_rate_of_change'] = (client['total_tx_q2'] - client['total_tx_q1']) / client['total_tx_q1']
```

> **`tx_q2q3_rate_of_change`**
```
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
```
client['avg_ss_duration'] = df_ss.groupby('CustomerId')['DurationMin'].mean()
```

> **`std_ss_duration`**
```
client['avg_ss_duration'] = df_ss.groupby('CustomerId')['DurationMin'].std()
```

> **`days_since_last_ss`**
```
client['days_since_last_ss'] = (CUTOFF_DATE - df_ss['SessionDate'].max()).dt.days
```

> **`ss_q1q2_rate_of_change`**
```
client['ss_q1q2_rate_of_change'] = (client['total_ss_q2'] - client['total_ss_q1']) / client['total_ss_q1']
```

> **`ss_q2q3_rate_of_change`**
```
client['ss_q2q3_rate_of_change'] = (client['total_tx_q3'] - client['total_tx_q2']) / client['total_tx_q2']
```

> **`failed_ratio_spike_q2`**
```
client['failed_ratio_spike_q2'] = (client['total_failed_ss_q2'] / client['total_ss_q2']) - (client['total_failed_ss_q1'] / client['total_ss_q1'])
```

> **`failed_ratio_spike_q3`**
```
client['failed_ratio_spike_q3'] = (client['total_failed_ss_q3'] / client['total_ss_q3']) - (client['total_failed_ss_q2'] / client['total_ss_q2'])
```

> **`failed_ratio_volatility`** 
```
client['failed_ratio_volatility'] = [(client['total_failed_ss_q1'] / client['total_ss_q1']), (client['total_failed_ss_q2'] / client['total_ss_q2']), (client['total_failed_ss_q3'] / client['total_ss_q3'])].std()
```



#### **Target**

| Variable      | Tipo de dato   | Descripción Funcional                                                                      | 
|---------------|----------------|--------------------------------------------------------------------------------------------| 
| `Exited`      | Int            | Condición de abando: 1 = Churn (Canceló) | 0 = No Churn (Permanencia)                      |



## 5. Resultados y conclusiones
---

* **Conclusiones más relevantes obtenidas a partir de los datos**
* **Resultado del proyecto: Breve descripción de la solución con imágenes explicativas**


## 6. Tecnologías utilizadas 🛠️

* `<Tech>`
* `<Tech>`
* `<Tech>`
* `<Tech>`


## 7. Agradecimientos 🤝

Presentar agradecimientos para Oracle, Alura, NoCountry y el programa ONE.


## 8. Desarrolladores del proyecto 👷

* **<Nombre>**
  - Rol:
* **<Nombre>**
  - Rol:
* **<Nombre>**
  - Rol: 
* **<Nombre>**
  - Rol:
* **<Nombre>**
  - Rol: