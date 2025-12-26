# **Fintech - Customer Churn**



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

### **Contexto del Negocio** 

El proyecto se centra en el producto de Préstamos por Convenio (Nómina) del "Banco Z" en Perú. Este producto se caracteriza por un bajo riesgo de morosidad (descuento por planilla), lo que convierte a estos clientes en perfiles altamente deseados por el sistema financiero. El problema principal es la Deserción Voluntaria por Compra de Deuda: la competencia ofrece mejores condiciones (menores tasas) para captar sus mejores clientes.

### **Objetivo** 

Reducir la pérdida de cartera vigente y aumentar el Lifetime Value (LTV) del cliente mediante la detección temprana de intenciones de prepago o compra de deuda.

### **La Solución** 

Hemos desarrollado un MVP End-to-End que integra un modelo de Machine Learning con una API REST funcional.

***Enfoque de Data Science:** Entrenamos un modelo de clasificación binaria utilizando datos históricos transaccionales, demográficos y del buró de crédito (RCC). Identificamos variables clave <VARIABLES> para evitar la deserción.

***Enfoque de Backend:*** Disponibilizamos el modelo a través de una API (Java/Spring Boot) que permite al negocio consultar el riesgo de un cliente en tiempo real, devolviendo una predicción clara ("Deserción" / "No Deserción") y su probabilidad asociada.



## 2. Acceso al proyecto 📂

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

<br><br><br>
Aquí podemos generalizar inicialmente a cada repositorio: Back-end, Front-end, Data Science. Y luego dentro de cada "sección explicar las etapas realizadas para
cada área.

<br><br><br>

## 4. Data Catalog (Catálogo de datos)

### [archivo](https://github.com/<link_hacia_el_dataset>)





#### **Clientes**

|Índice	| Variable	 | Definición Funcional	              |
|-------|------------|------------------------------------|
|0	    | Periodo	   | Mes de la toma de datos (YYYYMM).  | 
|1	    | CodCli	   | Identificador único del cliente.	  | 
|2	    | Edad	     | Edad del cliente.	                |
|3	    | Gener	     | Generación (Baby Boomer, X, Y, Z). |
|4	    | EstCiv	   | Estado Civil.                      | 
|5	    | Sexo	     | Género.                            |
|6	    | NZona	     | Zona geográfica/Sucursal .         |
|9	    | NivEduc	   | Nivel educativo.                   |


#### **Transacciones**

| Índice | Variable	  | Definición Funcional	                         |
|--------|------------|------------------------------------------------|
| 7	     | Ingreso	  | Ingreso mensual del cliente.	                 |
| 8	     | TipIng	    | Tipo: Dependiente vs Independiente.            |
| 10	   | RentSol	  | Renta declarada en la solicitud.	             |
| 11	   | AntiClie	  | Antigüedad bancaria (meses).	                 |
| 12	   | SectConv	  | Sector de la empresa (Salud, Educación, etc.). |
| 29	   | Activos	  | Patrimonio del cliente.	                       |


#### **Features: Características del Crédito**

| Índice | Variable	  | Definición Funcional	                                              |
|--------|------------|---------------------------------------------------------------------|
| 13	   | FechAprob  |	Fecha de aprobación.	                                              |
| 14	   | FechAper	  | Fecha de apertura/desembolso.	                                      |
| 15	   | FechVenc	  | Fecha de vencimiento.	                                              |
| 16	   | Desem	    | Monto original prestado.                                            |
| 17	   | TasaSol	  | Tasa de interés (TEA).                                              |
| 20	   | TipoOper	  | Nuevo vs Reenganche.	Reenganche = Cliente retenido anteriormente.  |
| 21	   | CuoPac	    | Plazo original (meses).	                                            |
| 22	   | CuoTot	    | Plazo total (incluye reprogramaciones).	                            |

#### **Features: Comportamiento Interno y Estado de Deuda**

| Índice   | Variable	    | Definición Funcional	                                            |
|----------|--------------|-------------------------------------------------------------------|
| 18	     | SalCap	      | Saldo Capital (Lo que debe hoy).	Fundamental.                    |
| 19	     | SalInt	      | Intereses pendientes.	                                            |
| 23	     | CuoPag	      | Cuotas pagadas.	Usar para crear ratio de avance (CuoPag/CuoTot).  |
| 24	     | CuoPen	      | Cuotas pendientes.	                                              |
| 25	     | NroReen	    | Cantidad de renovaciones históricas.                              |
| 26	     | FlagReEn	    | ¿El crédito actual es renovado?	                                  |
| 27	     | CantReprog	  | Cantidad de reprogramaciones.                                     |
| 30	     | CrossSell	  | Tenencia de otros productos.	                                    |

#### **Features: Sistema Financiero Externo (Competencia)**

| Índice	| Variable	   | Definición Funcional	                               |
|---------|--------------|-----------------------------------------------------|
| 28	    | Pasivos	     | Deuda total en todo el sistema.	                   |
| 31	    | SaldoRCC	   | Deuda reportada al regulador (Total).	             |
| 32	    | SaldoRCC_X   | Deuda con la Competencia (Otros bancos).	           |
| 33	    | MaxSalConv   | Máximo endeudamiento histórico.	                   |
| 34	    | PromSalConv  | Promedio endeudamiento histórico.	                 |
| 35	    | PromSowConv	 | Share of Wallet (% Deuda con nosotros).	           |
| 36	    | VarAnualConv | Variación de deuda anual.	                         |
| 37	    | CantCalifN	 | Meses con calificación "Normal".	                   |
| 38	    | PropCalifN	 | Proporción de calificación "Normal".	               |
| 39	    | PromCantEmp	 | Promedio histórico de entidades acreedoras.	       |
| 40	    | CantEmp	     | Cantidad actual de bancos con los que tiene deuda.	 |
| 41	    | Calif_Final	 | Calificación de riesgo (Normal, CPP, etc).          |


#### Target

| Índice | Variable                      | Descripción Funcional                                                                      | 
|--------|-------------------------------|--------------------------------------------------------------------------------------------| 
| 42     | `FlagDeser`                   | Condición de abando: 1 = Churn (Canceló) | 0 = No Churn (Permanencia)                       |




**Nota**: <aclración en caso de ser necesaria para las explicaciones siguientes>

Explicación del feature engineering para variables compuestas:

> **`variable_a_explicar`**
```
código o fórmula a partir de la cual se ha generado la variable
```

> **`variable_a_explicar`**
```
código o fórmula a partir de la cual se ha generado la variable
```

> **`variable_a_explicar`**
```
código o fórmula a partir de la cual se ha generado la variable
```

> **`variable_a_explicar`**
```
código o fórmula a partir de la cual se ha generado la variable
```


#### Features: <dataset origen si aplica>

| Feature               | Tipo                | Descripción                                                    | 
|-----------------------|---------------------|----------------------------------------------------------------| 
| `<Nombre Variable>`   | <Tipo Variable>     | <Breve descripción de la variable>                             |
| `<Nombre Variable>`   | <Tipo Variable>     | <Breve descripción de la variable>                             |
| `<Nombre Variable>`   | <Tipo Variable>     | <Breve descripción de la variable>                             |
| `<Nombre Variable>`   | <Tipo Variable>     | <Breve descripción de la variable>                             |
| `<Nombre Variable>`   | <Tipo Variable>     | <Breve descripción de la variable>                             |
| `<Nombre Variable>`   | <Tipo Variable>     | <Breve descripción de la variable>                             |
| `<Nombre Variable>`   | <Tipo Variable>     | <Breve descripción de la variable>                             |
| `<Nombre Variable>`   | <Tipo Variable>     | <Breve descripción de la variable>                             |

**Nota**: <aclración en caso de ser necesaria para las explicaciones siguientes>

Explicación del feature engineering para variables compuestas:

> **`variable_a_explicar`**
```
código o fórmula a partir de la cual se ha generado la variable
```

> **`variable_a_explicar`**
```
código o fórmula a partir de la cual se ha generado la variable
```

> **`variable_a_explicar`**
```
código o fórmula a partir de la cual se ha generado la variable
```

> **`variable_a_explicar`**
```
código o fórmula a partir de la cual se ha generado la variable
```


## 5. Resultados y conclusiones

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